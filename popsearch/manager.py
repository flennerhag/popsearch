"""Population based search
"""
import os
import time
from multiprocessing import Pool
from numpy.random import RandomState
from .state import State
from .eval import sample_job
from .sample import sample, perturb, build_params, pareto_cumulative
from .utils import get_logs, get_jid


###############################################################################

def initialize(rs, config, params):
    """Get a job id (i.e. seed) and force param"""
    force = rs.rand() < config.p_force
    jid = rs.randint(0, int(1e9))
    state = get_state(jid, force, rs, config, params)
    if not check_state(config, state):
        return initialize(rs, config, params)

    f = os.path.join(config.path, str(jid) + '.log')
    if os.path.exists(f):
        return initialize(rs, config, params)
    open(f, 'w').close()
    return state


def check_state(config, state):
    """Run a pre-check on a State instance"""
    if config.check_state is None:
        return True
    return config.check_state(state)


def get_state(jid, force, rs, config, params, force_sample=False):
    """Create a state instance for a jid"""
    if force_sample or rs.rand() > config.p_perturb:
        pars = sample(params)
    else:
        logs = get_logs(config.path)
        logs = sample_job(logs, 1, rs, *config.perturb_cdf)
        if len(logs) == 0:
            return get_state(jid, force, rs, config, params, force_sample=True)

        base = build_params(logs[0])
        pars = perturb(params, base)

    return State(jid, config, pars, force)


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
        files = get_logs(self.path)
        for f in files:
            with open(f, 'r') as _f:
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
        files = get_logs(self.path)
        n = 0
        v = 1e9
        j = None
        for f in files:
            with open(f, 'r') as _f:
                for l in _f:
                    if l.startswith('EVAL:'):
                        _, n_, v_ = l.split(':')
                        n_, v_ = int(n_), float(v_)
                        if v_ < v:
                            v = v_          # best score
                            j = get_jid(f)  # best score job id
                            n = n_          # best score eval step
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
        buffer (int): number of jobs per worker to keep in buffer, default=2
        sleep (float): seconds to wait before checking job completion, default=2
    """

    def __init__(
            self, callable, path, n_step, n_pop, n_job, p_perturb=0.8,
            min_eval_step=0, max_val=None, tolerance=None, check_state=None,
            p_force=0., force_n_step=None, perturb_cdf=None, eval_cdf=None,
            seed=None, n_eval_samples=100, buffer=2, sleep=0.1, plot=False):
        self.callable = callable
        self.path = path
        self.n_step = n_step
        self.n_pop = n_pop
        self.n_job = n_job
        self.p_perturb = p_perturb
        self.min_eval_step = min_eval_step
        self.max_val = max_val
        self.tolerance = tolerance
        self.check_state = check_state
        self.p_force = p_force
        self.buffer = buffer
        self.seed = seed
        self.n_eval_samples = n_eval_samples
        self.sleep = sleep
        self.plot = plot
        self.force_n_step = force_n_step if force_n_step else int(n_step/10)

        self.perturb_cdf = None
        self.eval_cdf = None
        self._set_cdf(perturb_cdf, eval_cdf)

    def _set_cdf(self, p_cdf, e_cdf):
        """Set the job perturb sampler cdf and the eval score sampler cdf."""

        def _set_cdf(f, arg):
            if f is None:
                f = (pareto_cumulative, arg)
            if isinstance(f, (float, int)):
                f = (pareto_cumulative, arg)
            if not isinstance(f, (tuple, list)):
                raise ValueError(
                    "Expected [p,e]_cdf=(f, args). Got {}".format(f))
            return f

        for i in zip(['perturb_cdf', 'eval_cdf'], [p_cdf, e_cdf], [1.0, 0.7]):
            name, cdf, arg = i
            tup = _set_cdf(cdf, arg)
            setattr(self, name, tup)


def get_async(results, plotter=None):
    """Clear completed jobs from results cache"""
    # TODO: plot completed jobs
    # TODO: Use callback in apply_async
    # Now, .get() will kill the entire popsearch job
    complete = []
    for i, (res, _) in enumerate(results):
        if res.ready():
            complete.append(i)

    for i in reversed(complete):
        res, jid = results.pop(i)
        res.get()
        if plotter is not None:
            plotter.plot(jid)


def run(config, params):
    """Job manager"""
    if config.seed is None:
        config.seed = int(time.time())
    rs = RandomState(config.seed)

    for par in params:
        if par.seed is None:
            par.reseed(rs.randint(0, int(1e6)))

    #######
    plotter = None
    if config.plot:
        from .visualize import Plotter
        # TODO: put plot_args in Config
        plotter = Plotter(config.path, config.plot == 'save', False, {}, {}, {})
    #######

    results = []
    pool = Pool(config.n_job)
    job = Job(config.path, config.n_step, config.n_pop)
    while not job.state():
        get_async(results, plotter)
        if len(results) <= config.buffer * config.n_job:
            state = initialize(rs, config, params)
            res = pool.apply_async(config.callable, (state,))
            results.append((res, state.jid))
        time.sleep(config.sleep)
