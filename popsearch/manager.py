"""Population based search
"""
import os
from multiprocessing import Pool
from numpy.random import RandomState
from .state import State
from .eval import find_leading_logs
from .sample import sample, perturb, build_params
from .sample import pareto_cumulative, inverse_transformation_sample


###############################################################################
# START JOB

def initialize(rs, config, params):
    """Get a job id (i.e. seed) and force param"""
    force = rs.uniform(0, 1) < config.p_force
    jid = rs.randint(0, int(1e6))

    file = os.path.join(config.path, str(jid) + '.log')
    if os.path.exists(file):
        return initialize(rs, config, params)
    open(file, 'w').close()

    return get_state(jid, force, rs, config, params)


def get_state(jid, force, rs, config, params, force_sample=False):
    """Create a state instance for a jid"""
    if force_sample or rs.rand() > config.p_perturb:
        pars = sample(params)
    else:
        logs = os.listdir(config.path)
        logs = [os.path.join(config.path, l) for l in logs]
        logs = sample_job(logs, n_samples=1, alpha=config.alpha, rs=rs)
        if len(logs) == 0:
            return get_state(jid, force, rs, config, params, force_sample=True)

        base = build_params(logs[0])
        pars = perturb(params, base)

    return State(jid, config, pars, force)


def sample_job(logs, n_samples, alpha, rs):
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
    _, vals, jids = find_leading_logs(logs, 0)
    if not vals:
        return vals

    svi = [tup for tup in sorted(zip(vals, jids))]
    vals = [tup[0] for tup in svi]
    ids = [tup[1] for tup in svi]

    cdf = pareto_cumulative(vals, alpha)
    samples = [
        ids[inverse_transformation_sample(cdf, rs)] for _ in range(n_samples)
    ]

    logs = [l for l in logs if int(l.split('/')[-1].split('.')[0]) in samples]
    return logs


###############################################################################
# MONITOR POPSEARCH

class Job(object):

    """Job manager
    """

    def __init__(self, path, n_step, n_pop):
        self.path = path
        self.n_step = n_step
        self.n_pop = n_pop
        self.best_jid = None
        self.best_val = None
        self.best_step = None
        self.complete = None

    def state(self):
        """Tracker for job completion."""
        new_changes = False

        # Check num completed jobs
        complete = 0
        files = os.listdir(self.path)
        for f in files:
            with open(os.path.join(self.path, f), 'r') as _f:
                for l in _f:
                    if l.startswith('EVAL:{}:'.format(self.n_step)):
                        complete += 1

        if complete != self.complete:
            new_changes = True
            self.complete = complete

        # Check best score
        jid, val, step = self.score()
        if self.best_jid != jid or self.best_val != val or self.best_step != step:
            new_changes = True
            self.best_jid = jid
            self.best_val = val
            self.best_step = step

        if new_changes:
            print("[STATUS] COMPLETE_RUNS={} LEADER_ID={} LEADER_SCORE={} "
                  "LEADER_STEP={}".format(complete, jid, val, step))

        # Return job status
        if complete == self.n_pop:
            return True
        return False

    def score(self):
        """Tracker for best score recorded"""
        files = os.listdir(self.path)
        n = 0
        v = 1e9
        j = None
        for f in files:
            with open(os.path.join(self.path, f), 'r') as _f:
                for l in _f:
                    if l.startswith('EVAL:'):
                        _, n_, v_ = l.split(':')
                        n_, v_ = int(n_), float(v_)
                        if v_ < v:
                            v = v_  # best score
                            j = int(f.split('.')[0])  # best score job id
                            n = n_  # best score eval step
        return j, v, n


###############################################################################
# RUN

class Config(object):

    """PopSearch configuration

    Args:
        callable (func): function to call during async loop
        path (str): path to log directory
        n_step (int): max number of eval steps
        n_pop (int): max number of jobs at n_step (termination criteria)
        p_force (float): probability of a force complete job, optional
        p_perturb (float): probability of sampling and perturbing job, optional
        alpha (float): shape parameter for perturbation sampling, optional
        max_val (int, float): a max eval value for force termination, optional
        seed (int): random seed, optional
    """

    __slots__ = [
        'callable', 'path', 'n_step', 'n_pop', 'n_job', 'max_val', 'seed',
        'p_force', 'p_perturb', 'alpha'
    ]

    def __init__(self, callable, path, n_step, n_pop, n_job, max_val=None,
                 p_force=0.95, p_perturb=0.5, alpha=1, seed=None):
        self.callable = callable
        self.path = path
        self.n_step = n_step
        self.n_pop = n_pop
        self.n_job = n_job
        self.max_val = max_val
        self.p_force = p_force
        self.p_perturb = p_perturb
        self.alpha = alpha
        self.seed = seed


def run(config, params):
    """Job manager"""
    if config.seed is None:
        from time import time
        config.seed = int(time())
    rs = RandomState(config.seed)

    for par in params:
        if par.seed is None:
            par.reseed(rs.randint(0, int(1e6)))

    pool = Pool(config.n_job)
    job = Job(config.path, config.n_step, config.n_pop)
    while not job.state():
        state = initialize(rs, config, params)
        pool.apply_async(config.callable, (state,))
