"""Eval routines

Functions for evaluating state of job against cache.
"""
import os
from numpy import percentile
from numpy.random import RandomState
from .sample import samplePerturbed


def build_params(params):
    out = {}
    params = params.split(',')
    for par in params:
        key, ptype, val = par.split('-')

        if 'int' in ptype:
            val = int(val)
        elif 'float' in ptype:
            val = float(val)

        out[key] = val
    return out


def read_log(log, step, values, params):
    """read a log file against a chid"""
    lval = None
    lpar = None
    for line in log:
        if line.startswith('PARAMS:'):
            lpar = line.split(':')[1].replace('\n', '')
        if line.startswith('EVAL:{}'.format(step)):
            lval = line.split(':')[2].replace('\n', '')
            lval = float(lval)

    if lval is not None:
        values.append(lval)
        params.append(build_params(lpar))


def extract_logs(step, logs):
    """Extract values and params from all logs"""
    values, params = [], []
    for log in logs:
        with open(log, 'r') as l:
            read_log(l, step, values, params)
    return values, params


def eval_step(path, val, step, seed=None):
    """Evaluate a job at given checkpoint"""
    logs = os.listdir(path)
    logs = [os.path.join(path, l) for l in logs]
    vals, params = extract_logs(step, logs)

    if not vals:
        # Too early - continue
        return 1

    perc = percentile(vals, [10, 80])

    if val <= perc[0]:
        # Top 40 %: continue
        return 1

    if val >= perc[1]:
        # Bottom 20 %: complete resample
        return 0

    # Mid 20-60 %
    # Copy and perturb from top 40 %
    rs = RandomState(seed)
    ppars = [p for i, p in enumerate(params) if vals[i] < val]
    pix = rs.randint(0, len(ppars))
    ppars = ppars[pix]
    return samplePerturbed(ppars, seed=seed)
