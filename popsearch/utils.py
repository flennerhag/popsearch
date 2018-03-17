"""Utility functions
"""
import os
from numpy import isnan, isinf, inf


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


###############################################################################

def get_logs(p):
    """Get log files from path"""
    return [os.path.join(p, l) for l in os.listdir(p) if l.endswith('.log')]


def get_jid(log_name):
    """Strip path and file type from the log name to return the jid as int"""
    return int(log_name.split('/')[-1].split('.')[0])


def jid_to_log(jids, path="", logs=None):
    """Construct log files from a list of jids, or larger set of logs"""
    if not logs:
        return [os.path.join(path, "{}.log".format(jid)) for jid in jids]
    return [l for l in logs if get_jid(l) in jids]


def sort_lists(*lists):
    """Sort a list of lists based on the first list"""
    out = [[] for _ in range(len(lists))]
    for vals in sorted(zip(*lists)):
        for i, val in enumerate(vals):
            out[i].append(val)
    return tuple(out)


def eval_lines(log, at_step=None):
    """Generator for eval lines in a log file"""
    criterion = 'EVAL:' if at_step is None else 'EVAL:{}:'.format(at_step)
    with open(log, 'r') as open_log:
        for line in open_log:
            if line.startswith(criterion):
                _, step, val = line.split(':')
                yield int(step), float(val)
                if at_step is not None:
                    break
    yield None, None


def read_logs_at_step(logs, step):
    """Read a log file at a given eval step"""
    values = []
    for log in logs:
        _, val = eval_lines(log, step).__next__()
        if check_val(val):
            values.append(val)
    return values


def find_best_step(logs, min_step, return_val=False):
    """Find current best checkpoint step"""
    val = inf
    step = None
    for log in logs:
        log_step = None
        log_val = inf
        for ls, lv in eval_lines(log):
            if check_val(lv) and lv < log_val:
                log_val = lv
                log_step = ls

        if log_val < val and log_step >= min_step:
            val = log_val
            step = log_step

    if return_val:
        return step, val
    return step


def find_leading_logs(logs, min_step):
    """Find the checkpoint step with best loss and return jids and loss vals"""
    best_vals, best_jids, best_logs = [], [], []
    best_step = find_best_step(logs, min_step, return_val=False)

    if best_step:
        for log in logs:
            log_id = get_jid(log)
            log_step, log_val = eval_lines(log, best_step).__next__()
            if check_val(log_val):
                best_vals.append(log_val)
                best_jids.append(log_id)
                best_logs.append(log)

    return best_step, best_vals, best_jids, best_logs
