"""Eval routines

Functions for evaluating a job against against current state of popsearch.
"""
from numpy import isnan, isinf, inf
from .sample import inverse_transformation_sample
from .utils import get_logs, jid_to_log


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
        if line.startswith('EVAL:{}:'.format(step)):
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


def extract_logs(step, logs):
    """Extract parameters and performance at eval step from sets of logs"""
    # Read values and parameters at STEP of most advanced jobs
    values = []
    for log in logs:
        with open(log, 'r') as l:
            read_log(l, step, values)
    return values


def sort_lists(*lists):
    """Sort a list of lists based on the first list"""
    out = [[] for _ in range(len(lists))]
    for vals in sorted(zip(*lists)):
        for i, val in enumerate(vals):
            out[i].append(val)
    return tuple(out)


###############################################################################

def sample_cdf(cdf_vals, sample_vals, n_samples, rs, cdf, *cdf_args):
    """Sample a cdf with given vals"""
    sampled_cdf = cdf(cdf_vals, *cdf_args)
    samples = [sample_vals[inverse_transformation_sample(sampled_cdf, rs)]
               for _ in range(n_samples)]
    return samples


def sample_job(logs, n_samples, rs, cdf, *cdf_args):
    """Sample a log file from current job state

    :func:`sample_job` uses a discrete approximation of the pareto distribution
    to sample a log file from the subset of log files at the current best
    checkpoint step.

    Args:
        logs (list): list of log files paths (full path)
        n_samples (int): number of samples requested
        alpha (float): shape parameter, 1 is a good default
        rs (RandomState): random state engine

    Returns:
        samples (list): list of sampled log files from the logs input list
    """
    _, vals, jids = find_leading_logs(logs, 1)
    if not vals:
        return vals

    vals, jids = sort_lists(vals, jids)
    samples = sample_cdf(vals, jids, n_samples, rs, cdf, *cdf_args)
    return jid_to_log(samples, logs=logs)


def eval_step(path, val, step, rs, min_step, n_samples, cdf, *cdf_args):
    """Evaluate a job at given checkpoint"""
    if isnan(val) or isinf(val):
        return 0

    if min_step is None:
        min_step = step
    min_step = max(min_step, step)

    logs = get_logs(path)
    best_step, best_vals, best_jids = find_leading_logs(logs, min_step)
    if not best_vals:
        return 1

    logs = jid_to_log(best_jids, path)
    vals = extract_logs(step, logs)
    best_vals, vals = sort_lists(best_vals, vals)
    samples = sample_cdf(best_vals, vals, n_samples, rs, cdf, *cdf_args)

    from numpy import std, mean
    v = sum(samples) / n_samples
    x = max(0, best_step - step) / best_step
    modifier = 1 + (1.5 * x) ** 4
    threshold = modifier * v
#    print(val, v, std(samples), mean(best_vals), mean(vals), threshold, modifier, best_step, step)
    if val <= threshold:
        return 1
    return 0
