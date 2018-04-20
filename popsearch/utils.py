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


def parse_params(log):
    params = None
    with open(log, 'r') as log:
        for line in log:
            if line.startswith('PARAMS:'):
                params = line.split(':')[1].split(';')
                break

    assert params is not None, "No parameters found in log file"
    return params


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
        log_step, log_val = get_score(log, min_step, best=True)
        if not check_val(log_val):
            continue
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


def get_score(log, min_step, best=True):
    """Find the best score in a log file"""

    def criterion(val, benchmark):
        if not check_val(val):
            return False

        if benchmark is None:
            return True

        if best:
            if val < benchmark:
                return True
            return False
        if val < benchmark:
            return False
        return True

    best_step = None
    best_val = None
    for line_step, line_val in eval_lines(log):
        if line_step is None or line_step < min_step:
            continue
        if criterion(line_val, best_val):
            best_val = line_val
            best_step = line_step
    return best_step, best_val


def get_best_scores(logs, min_step):
    """Find the checkpoint step with best loss and return jids and loss vals"""
    best_step, best_val, best_log = [], [], []
    for log in logs:
        ls, lv = get_score(log, min_step, best=True)
        if ls is not None and check_val(lv):
            best_step.append(ls)
            best_val.append(lv)
            best_log.append(log)
    return best_step, best_val, best_log
