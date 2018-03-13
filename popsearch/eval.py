"""Eval routines

Functions for evaluating a job against against current state of popsearch.
"""
import os
from numpy import ceil, isnan, isinf, inf


###############################################################################

def check_val(val):
    """Check that a value is not None, NaN or Inf"""
    return val is not None and not (isnan(val) or isinf(val))


def test_val(lval, cvals):
    """Wrapper around taking the min of a possibly empty list"""
    if not cvals:
        return True
    else:
        return lval < min(cvals)


def check_line_eval(line):
    """Check a line in a log for the EVAL: flag and return status"""
    ls, lv = None, None
    if line.startswith('EVAL:'):
        _, ls, lv = line.split(':')
        ls = int(ls)
        lv = float(lv.replace('\n', ''))
    return ls, lv


def read_log(log, step, values):
    """Read a log file at a given eval step"""
    lval = None
    for line in log:
        if line.startswith('EVAL:{}'.format(step)):
            lval = line.split(':')[2].replace('\n', '')
            lval = float(lval)

    if lval is not None and not isnan(lval) and not isinf(lval):
        values.append(lval)


def find_best_step(logs, min_step):
    """Find current best checkpoint step"""
    val = inf
    step = None
    for log in logs:
        with open(log, 'r') as l:
            log_step = None
            log_val = inf
            for line in l:
                ls, lv = check_line_eval(line)
                if check_val(lv) and lv < log_val:
                    log_val = lv
                    log_step = ls

        if log_val < val and log_step >= min_step:
            val = log_val
            step = log_step
    return step


def find_leading_logs(logs, min_step):
    """Find the checkpoint step with best loss and return jids and loss vals"""
    step = find_best_step(logs, min_step)

    vals, ids = [], []
    for log in logs:
        with open(log, 'r') as l:
            log_id = int(l.name.split('/')[-1].split('.')[0])
            for line in l:
                log_step, log_val = check_line_eval(line)
                if check_val(log_val) and log_step == step:
                    vals.append(log_val)
                    ids.append(log_id)

    return step, vals, ids


def prune_logs(logs, min_step):
    """Return top x% of logs in leading checkpoint step"""
    step, vals, ids = find_leading_logs(logs, min_step)

    if not vals:
        return vals

    # Select top x% of jobs in the best eval step
    s = min(len(vals), int(ceil(len(vals) * 0.5)))
    top_ids = [b for a, b in sorted(zip(vals, ids))][:s]
    top_logs = [l for l in logs if int(l.split('/')[-1].split('.')[0]) in top_ids]
    return top_logs


def extract_logs(step, logs):
    """Extract parameters and performance at eval step from sets of logs"""
    # Read values and parameters at STEP of most advanced jobs
    values = []
    for log in logs:
        with open(log, 'r') as l:
            read_log(l, step, values)
    return values


###############################################################################

def eval_step(path, val, step):
    """Evaluate a job at given checkpoint"""
    if isnan(val) or isinf(val):
        # Divergent job: cut
        return 0

    # Select logs of the current best population
    # That is, find the best val, at some eval step,
    # and select the top 20% logs from that eval step
    logs = os.listdir(path)
    logs = [os.path.join(path, l) for l in logs if l.endswith('.log')]
    logs = prune_logs(logs, step)

    # Check if current job in its current eval step
    # is comparable to, or better than, the evals of the
    # selected top logs at the current job's eval step
    vals = extract_logs(step, logs)
    if not vals:
        # Too early evaluation: continue job
        return 1

    threshold = max(vals)
    if val <= threshold:
        # Current job is comparable to or better than what the
        # top 20% configs were at the current job's eval step: continue
        return 1

    return 0
