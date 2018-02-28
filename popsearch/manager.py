"""Population based search
"""
import os
from multiprocessing import Pool
from numpy.random import RandomState

from .sample import sample


def init_job(path, rs):
    """Get a job id (i.e. seed) and force param"""
    # Sample jid and force variables
    force = rs.uniform(0, 1) > 0.95
    jid = rs.randint(0, 1e6)

    # Check for file overlap and claim
    file = os.path.join(path, str(jid) + '.log')
    if os.path.exists(file):
        return init_job(path, rs)
    open(file, 'w').close()

    return jid, force


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
                    if l.startswith('EVAL:{}:'.format(self.n_step - 1)):
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


def get_iterpars(iterpars, params, seed=None):
    """Sample iterpar from cache of iterpars"""
    n = len(iterpars)
    if n == 0:
        iterpar = None
    else:
        i = RandomState(seed + n).randint(0, n)
        iterpar = iterpars.pop(i)
    return check_iterpars(iterpar, params, seed)


def check_iterpars(iterpar, params, seed=None):
    """Check iterpars and resample if necessary"""
    if iterpar is None or iterpar == 0:
        return {par: sample(args, seed=seed) for par, args in params.items()}

    for k, v in params.items():
        mn = mx = rng = None
        if len(v) == 2:
            _, rng = v
        elif len(v) == 3:
            _, mn, mx = v
        else:
            raise ValueError("params not properly specified")

        if rng is not None:
            if not iterpar[k] in rng:
                iterpar[k] = sample(v, seed=seed)
        else:
            if (iterpar[k] > mx) or (iterpar[k] < mn):
                iterpar[k] = sample(v, seed=seed)

    return iterpar


def get_async(results):
    """Clear completed jobs from results cache"""
    out, complete = [], []
    for i in range(len(results)):
        if results[i].ready():
            complete.append(i)

    for i in reversed(complete):
        out.append(results.pop(i).get())

    return out


def run(config):
    """Job manager"""
    call = config['call']
    path = config['path']
    max_val = config['max_val']
    params = config['params']
    n_step = config['n_step']
    n_pop = config['n_pop']
    n_jobs = config['n_job']

    # Seed job
    if 'seed' in config:
        seed = config['seed']
    else:
        from time import time
        seed = int(time())
    rs = RandomState(seed)

    iterpars = []          # Cache of cleared results
    results = []           # Cache of uncleared results
    pool = Pool(n_jobs)    # Worker pool
    job = Job(path, n_step, n_pop)
    while not job.state():
        # Get unique job id : also seed for param sampling
        jid, force = init_job(path, rs)

        # Get params from batch of previous jobs
        iterpar = get_iterpars(iterpars, params, jid)

        # Run jobs
        job_args = (jid, path, force, max_val)
        res = pool.apply_async(call, (job_args, iterpar))
        results.append(res)

        # Update cache of cleared results
        iterpars.extend(get_async(results))
