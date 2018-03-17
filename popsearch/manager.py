"""Population based search
"""
import os
import time
from multiprocessing import Pool
from numpy.random import RandomState
from .state import State
from .sample import sample, pareto_cumulative
from .rules import job_sample, eval_sample, eval_predict, eval_double
from .utils import get_logs, get_jid, eval_lines, check_val


###############################################################################

def initialize(rs, config, params, perturb):
    """Get a job id (i.e. seed) and force param"""
    force = rs.rand() < config.p_force
    jid = rs.randint(0, int(1e9))
    state = get_state(jid, force, rs, config, params, perturb)
    if not check_state(config, state):
        return initialize(rs, config, params, perturb)

    f = os.path.join(config.path, str(jid) + '.log')
    if os.path.exists(f):
        return initialize(rs, config, params, perturb)
    open(f, 'w').close()
    return state


def check_state(config, state):
    """Run a pre-check on a State instance"""
    if config.check_state is None:
        return True
    return config.check_state(state)


def get_state(jid, force, rs, config, params, perturb, force_sample=False):
    """Create a state instance for a jid"""
    if force_sample or not perturb(rs):
        pars = sample(params)
    else:
        logs = get_logs(config.path)

        f, *args = config.perturb_rule
        pars = f(logs, rs, params, *args)
        if pars is None:
            return get_state(jid, force, rs, config, params, perturb,
                             force_sample=True)

    return State(jid, config, pars, force)


###############################################################################
# MONITOR POPSEARCH
class Perturb(object):

    def __init__(self, min_p, decay, p_reset):
        self.min_p = min_p
        self.decay = decay
        self.p_reset = p_reset
        self._p_perturb = min_p

    def __call__(self, rs):
        p_perturb = rs.rand()
        perturb = p_perturb <= self._p_perturb

        if perturb:
            self._p_perturb = max(self.min_p, self._p_perturb * self.decay)

        if self._p_perturb == self.min_p:
            p_reset = rs.rand()
            if p_reset <= self.p_reset:
                self._p_perturb = 1.0

        print(perturb, p_perturb, self._p_perturb)
        return perturb

    def reset(self):
        self._p_perturb = 1.0


class Job(object):

    """Job manager
    """

    def __init__(self, path, n_step, n_pop, perturb):
        self.path = path
        self.n_step = n_step
        self.n_pop = n_pop
        self.perturb = perturb
        self.best_jid = None
        self.best_val = None
        self.best_step = None
        self.complete = None

    def state(self):
        """Tracker for job completion."""
        complete = 0
        files = get_logs(self.path)
        for f in files:
            with open(f, 'r') as _f:
                for l in _f:
                    if l.startswith('EVAL:{}:'.format(self.n_step)):
                        complete += 1
        self.complete = complete

        # Check best score
        jid, val, step = self.score()
        new_jid = self.best_jid != jid
        new_val = self.best_val != val
        new_step = self.best_step != step
        new_changes = new_val or new_jid or new_step
        if new_changes:
            self.best_jid = jid
            self.best_val = val
            self.best_step = step
            print("[STATUS] COMPLETE_RUNS={} LEADER_ID={} LEADER_SCORE={} "
                  "LEADER_STEP={}".format(complete, jid, val, step))

        if new_val and new_jid:
            # Don't reset perturb only because val step changes - want new jid
            self.perturb.reset()

        if complete == self.n_pop:
            return True
        return False

    def score(self):
        """Tracker for best score"""
        logs = get_logs(self.path)
        step, val, jid = None, None, None
        for log in logs:
            for iter_step, iter_val in eval_lines(log):
                if val is None or (check_val(iter_val) and iter_val < val):
                    jid = get_jid(log)  # best score job id
                    val = iter_val      # best score
                    step = iter_step    # best score eval step
        return jid, val, step


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

    def __init__(self,
                 callable,
                 path,
                 n_step,
                 n_pop,
                 n_job,
                 perturb=(0.2, 0.9, 0.05),
                 min_eval_step=0,
                 max_val=None,
                 tolerance=None,
                 check_state=None,
                 p_force=0.,
                 force_n_step=None,
                 perturb_rule=('sample', 'pareto', (1.0,)),
                 eval_rule=('sample', 'pareto', (1.0,)),
                 seed=None,
                 n_eval_samples=100,
                 buffer=2,
                 sleep=0.1,
                 plot=False,
                 plot_config=None):
        self.callable = callable
        self.path = path
        self.n_step = n_step
        self.n_pop = n_pop
        self.n_job = n_job
        self.perturb = perturb
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
        self.force_n_step = force_n_step if force_n_step else int(n_step / 10)

        self.perturb_rule = None
        self.eval_rule = None
        self.plot_config = None
        self._set_rules(perturb_rule, eval_rule)
        self._set_plot_config(plot, plot_config)

    def _set_rules(self, pt, ev):
        """Set the job perturb sampler cdf and the eval score sampler cdf."""

        def get_default(tup, default_rule):
            if tup[0] == 'sample':
                f = tup[1]
                a = tup[2]
                if f == 'pareto':
                    f = pareto_cumulative
                return default_rule[0], f, a
            if tup[0] == 'predict':
                a = tup[1]
                return default_rule[1], a
            if tup[0] == 'double':
                f = pareto_cumulative if tup[1] is None else tup[1]
                a = (1.0,) if tup[2] is None else tup[2]
                w = int(self.n_step / 10) if tup[3] is None else tup[3]
                return default_rule[2], f, a, w
            return tup

        self.perturb_rule = get_default(pt, [job_sample])
        self.eval_rule = get_default(ev, [eval_sample, eval_predict, eval_double])

    def _set_plot_config(self, plot, plot_config):
        """Set plot configuration if applicable"""
        if not plot:
            return

        default_config = {
            'save': False,
            'semilogy': False,
            'fig_kwargs': {},
            'plot_kwargs': {},
            'save_kwargs': {},
        }

        if plot_config is None:
            self._plot_config = default_config
            return

        for k, v in default_config.items():
            if k not in plot_config:
                plot_config[k] = v

        self.plot_config = plot_config


def get_async(results, plotter=None):
    """Clear completed jobs from results cache"""
    # TODO: callback in apply_async to prevent .get() from killing parent proc?
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
            par.reseed(rs.randint(0, int(1e9)))

    #######
    plotter = None
    if config.plot:
        from .visualize import Plotter
        # TODO: put plot_args in Config
        plotter = Plotter(config.path, **config.plot_config)
    #######

    results = []
    pool = Pool(config.n_job)
    perturb = Perturb(*config.perturb)
    job = Job(config.path, config.n_step, config.n_pop, perturb)
    while not job.state():
        get_async(results, plotter)
        if len(results) <= config.buffer * config.n_job:
            state = initialize(rs, config, params, perturb)
            res = pool.apply_async(config.callable, (state,))
            results.append((res, state.jid))
        time.sleep(config.sleep)
