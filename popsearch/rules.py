"""Default sampling and decision rules"""
from .sample import sample_cdf, build_params, perturb
from .utils import (find_best_step, find_leading_logs, get_logs,
                    read_logs_at_step, sort_lists, check_val)


###############################################################################

def job_sample(logs, rs, params, cdf, cdf_args):
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
    _, vals, _, logs = find_leading_logs(logs, 1)
    if not vals:
        return None

    vals, logs = sort_lists(vals, logs)
    logs = sample_cdf(vals, logs, 1, rs, cdf, *cdf_args)
    base = build_params(logs[0])
    pars = perturb(params, base)
    return pars


###############################################################################

def eval_sample(state, cdf, cdf_args):
    """Evaluate a job at given checkpoint"""
    rs = state.rs
    path = state.path
    step = state.step
    val = state.values[-1]
    min_step = state.min_eval_step
    n_samples = state.config.n_eval_samples

    if not check_val(val):
        return 0

    if min_step is None:
        min_step = step
    min_step = max(min_step, step)

    logs = get_logs(path)
    best_step, best_vals, best_jids, best_logs = find_leading_logs(logs, min_step)
#    print(min_step, best_step, best_vals, best_jids, best_logs)
    if not best_vals:
        return 1

    vals = read_logs_at_step(best_logs, step)
    best_vals, vals = sort_lists(best_vals, vals)
    samples = sample_cdf(best_vals, vals, n_samples, rs, cdf, *cdf_args)

    v = sum(samples) / n_samples
    x = max(0, best_step - step) / best_step
    modifier = 1 + (1.5 * x) ** 4
    threshold = modifier * v

    if val <= threshold:
        return 1
    return 0


def eval_predict(state, window_size):
    """Linear exterpolation of job to best_checkpoint"""
    if len(state.values) < window_size:
        return 1

    from numpy.linalg import lstsq

    curr_step = state.step
    hist_step = curr_step - window_size
    max_step = state.config.n_step
    x = [[1, i] for i in range(1 + hist_step, curr_step + 1)]
    y = state.values[-window_size:]

    coef, _, _, _ = lstsq(x, y)

    logs = get_logs(state.path)
    _, v = find_best_step(logs, state.config.min_eval_step, return_val=True)
    pred = coef[0] + coef[1] * max_step

    return int(1 * (pred <= v))


def eval_double(state, cdf, cdf_args, window_size):
    """Run a double eval, starting with predict and then sampling"""
    if eval_predict(state, window_size):
        return eval_sample(state, cdf, cdf_args)
    return 0
