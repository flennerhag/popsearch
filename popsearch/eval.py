"""Eval routines

Functions for evaluating state of job against cache.
"""
import os
from numpy import ceil, isnan, isinf
from numpy.random import RandomState
from .sample import samplePerturbed


def build_params(params):
    """Build a parameter dictionary out of a params string"""
    out = {}
    params = params.split(';')
    for par in params:
        key, ptype, val = par.split(',')
        if 'int' in ptype:
            val = int(val)
        elif 'float' in ptype:
            val = float(val)
        out[key] = val
    return out


def scan_log(log, curr_step, curr_vals, curr_ids):
    """Scan log for best eval step"""
    log_step = 0
    log_val = 1e9
    log_id = int(log.name.split('/')[-1].split('.')[0])
    for line in log:
        if line.startswith('EVAL:'):
            _, log_step, log_val = line.split(':')
            log_step = int(log_step)
            log_val = float(log_val.replace('\n', ''))

    if not curr_vals or (log_step >= curr_step and (log_val < min(curr_vals))):
        # New best eval step: reset counter
        curr_step = log_step
        curr_vals[:] = []
        curr_ids[:] = []

    if log_step == curr_step and (not isnan(log_val) or not isinf(log_val)):
        # Add log to lists
        curr_ids.append(log_id)
        curr_vals.append(log_val)


def read_log(log, step, values, params):
    """Read a log file at a given eval step"""
    lval = None
    lpar = None
    for line in log:
        if line.startswith('PARAMS:'):
            lpar = line.split(':')[1].replace('\n', '')

        if line.startswith('EVAL:{}'.format(step)):
            lval = line.split(':')[2].replace('\n', '')
            lval = float(lval)

    if lval is not None and not isnan(lval) and not isinf(lval):
        values.append(lval)
        params.append(build_params(lpar))


def prune_logs(logs, min_step):
    """Split logs into good, bad and ugly based on current best checkpoint"""
    # Find step with best loss
    step, vals, ids = min_step, [], []
    for log in logs:
        with open(log, 'r') as l:
            scan_log(l, step, vals, ids)

    if not vals:
        return vals

    # Select top x% of jobs in the best eval step
    s = min(len(vals), int(ceil(len(vals) * 0.2)))
    top_ids = [b for a, b in sorted(zip(vals, ids))][:s]
    top_logs = [l for l in logs if int(l.split('/')[-1].split('.')[0]) in top_ids]
    return top_logs


def extract_logs(step, logs):
    """Extract parameters and performance at eval step from sets of logs"""
    # Read values and parameters at STEP of most advanced jobs
    values, params = [], []
    for log in logs:
        with open(log, 'r') as l:
            read_log(l, step, values, params)
    return values, params


def eval_step(path, val, step, seed=None):
    """Evaluate a job at given checkpoint"""
    if isnan(val) or isinf(val):
        # Divergent job: cut
        return 0

    # Select logs of the current best population
    # That is, find the best val, at some eval step,
    # and select the top 20% logs from that eval step
    logs = os.listdir(path)
    logs = [os.path.join(path, l) for l in logs]
    logs = prune_logs(logs, step)

    # Check if current job in its current eval step
    # is comparable to, or better than, the evals of the
    # selected top logs at the current job's eval step
    vals, params = extract_logs(step, logs)
    if not vals:
        # Guard against too early evaluation: continue job
        return 1

    threshold = max(vals)
    if val <= threshold:
        # Current job is comparable to or better than what the
        # top 20% configs were at the current job's eval step: continue
        return 1

    # Else: restart
    # Need to determine whether to exploit or explore
    rs = RandomState(seed)
    if rs.uniform(0, 1) > 0.6:
        # 20% of restarts are purely random
        return 0

    # 80 % of restarts are perturbations from top 20 %
    pix = rs.randint(0, len(params))
    ppars = params[pix]
    return samplePerturbed(ppars, seed=seed + pix)
