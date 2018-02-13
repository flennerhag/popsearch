"""
Routines for sampling parameters
"""
# pylint: disable=missing-docstring
from numpy.random import RandomState

INT_PERTURB_MIN = 1
INT_PERTURB_MAX = 1
FLOAT_PERTURB_MIN = 0.1
FLOAT_PERTURB_MAX = 0.1


def uniform(mn, mx, rs):
    """Uniform dist in [mn, mx)"""
    return (mx - mn) * rs.rand() + mn


def sampleDiscrete(rng, rs):
    """Sample from list"""
    return rng[rs.randint(0, len(rng))]


def sampleBounded(ptype, mn, mx, rs):
    if ptype is int:
        return rs.randint(mn, mx + 1)
    if ptype is float:
        return uniform(mn, mx, rs)
    else:
        raise NotImplementedError("Parameter type not understood")


def sample(args, seed=None):
    """Sample parameter"""
    rs = RandomState(seed=seed)
    if len(args) not in [2, 3]:
        raise ValueError("Arguments should be tuple of two's or three's")

    if len(args) == 2:
        ptype, rng = args
        assert isinstance(rng, list), "Range variable is not list"
        mn = mx = None

    if len(args) == 3:
        ptype, mn, mx = args
        rng = None

    if ptype is str:
        assert rng, "String param requires range"

    return sampleDiscrete(rng, rs=rs) if rng else sampleBounded(ptype, mn, mx, rs=rs)


def samplePerturbed(params, seed=None):
    """return sample of perturbed parameters"""
    ppar = {}
    for k, v in params.items():
        ptype = type(v)
        if ptype is int:
            mn = v - INT_PERTURB_MIN
            mx = v + INT_PERTURB_MAX
            ppar[k] = sample((ptype, mn, mx), seed=seed)
        if ptype is float:
            mn = v * (1 - FLOAT_PERTURB_MIN)
            mx = v * (1 + FLOAT_PERTURB_MAX)
            ppar[k] = sample((ptype, mn, mx), seed=seed)
        else:
            # String parameters are ignored
            # These typically represent a case variable
            # cannot be perturbed
            ppar[k] = v
    return ppar
