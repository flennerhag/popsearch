"""Population based search
"""
import os
from multiprocessing import Pool
from .sample import sample
# pylint: disable=missing-docstring


def get_jid(path):
    logs = os.listdir(path)
    jids = [int(l.split(':')[0]) for l in logs]
    if jids:
        return max(jids) + 1
    return 0


def job_complete(path, n_step, n_pop):
    """Tracker for job completion."""
    files = os.listdir(path)
    complete = 0
    for f in files:
        with open(os.path.join(path, f), 'r') as _f:
            for l in _f:
                if l.startswith('EVAL:{}:'.format(n_step - 1)):
                    complete += 1

    status = job_status(path)
    print("[STATUS] COMPLETE_RUNS={} ".format(complete + 1) + status)

    if (complete + 1) == n_pop:
        return True
    return False


def job_status(path):
    files = os.listdir(path)
    n = 0
    v = 1e9
    b = None
    for f in files:

        last_evl = None
        with open(os.path.join(path, f), 'r') as _f:
            for l in _f:
                if l.startswith('EVAL:'):
                    last_evl = l
        if last_evl:
            _, n_, v_ = last_evl.split(':')
            n_, v_ = int(n_), float(v_)
            if (n_ >= n) and (v_ < v):
                n = int(n_)
                v = v_
                b = f
    if b:
        return "LEADER_ID={} LEADER_SCORE={} STEP_COUNT={}".format(
            b.split(':')[0], v, n + 1)
    return ""


def check_iterpars(iterpar, params, seed=None):
    """Check iterpars and resample if necessary"""
    if iterpar is None:
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


def run(config):
    """Job manager"""
    call = config['call']
    path = config['path']
    params = config['params']
    n_step = config['n_step']
    n_pop = config['n_pop']
    n_jobs = config['n_job']

    jid = None
    iterpars = None

    pool = Pool(n_jobs)
    while not job_complete(path, n_step, n_pop):
        jid = get_jid(path)
        iterpars = check_iterpars(iterpars, params, jid)
        iterpars = pool.apply_async(call, (jid, path, iterpars)).get()
