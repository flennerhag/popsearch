"""
Routines for sampling parameters
"""
from numpy.random import RandomState


###############################################################################

PERTURB_RANGE = {
    int: (1, 1),
    float: (0.1, 0.1),
    bool: False,
    str: False
}


def uniform(mn, mx, rs):
    """Sample from U[mn, mx)"""
    return (mx - mn) * rs.rand() + mn


###############################################################################
# SAMPLE PARAMS

def sample(params):
    """Sample parameters"""
    out = {}
    for par in params:
        if not isinstance(par, Parameter):
            # Nested sampling
            par, pparams = par

            ps = par.sample()
            pparams = pparams[ps]

            out[par.name] = ps
            out.update(sample(pparams))
        else:
            out[par.name] = par.sample()
    return out


def perturb(params, values):
    """Generate a perturbation of a params dict"""
    out = {}
    for par in params:
        if not isinstance(par, Parameter):
            # Nested sampling
            par, pparams = par

            val = values[par.name]
            ps = par.perturb(val)
            pparams = pparams[ps]

            out[par.name] = ps
            out.update(perturb(pparams, values))
        else:
            val = values[par.name]
            out[par.name] = par.perturb(val)
    return out


def get_type(type_name, value):
    """Convert a string value to correct type based on a type specification"""
    if 'int' in type_name:
        return int(value)
    if 'float' in type_name:
        return float(value)
    if 'bool' in type_name:
        return 'True' in value
    if 'str' in type_name:
        return value
    raise ValueError("Type format not understood")


def build_params(log_path):
    """Build a parameter dictionary from a log file."""
    params = None
    with open(log_path, 'r') as log:
        for line in log:
            if line.startswith('PARAMS:'):
                params = line.split(':')[1].split(';')
                break

    assert params is not None, "No parameters found in log file"

    out = {}
    for par in params:
        pname, ptype, pvalue = par.split(',')
        pvalue = get_type(ptype, pvalue)
        out[pname] = pvalue
    return out


def pareto_cumulative(values, alpha=1):
    """Cumulative Pareto Distribution"""
    xm = min(values)
    values = [1 - (xm / v) ** alpha for v in values]
    return values


def inverse_transformation_sample(cdf, rs):
    """Inverse Transformation Sampling of Discrete Observations

    This little hack deploys inverse transformation sampling to
    a continuouos distribution, of which we have a discrete number
    of points to actually sample from. We use the ITS cutoff rule

    F_X(x) <= u

    to tell us which observations we can randomly sample from. Because
    ITS adheres to F_X, so does sampling, but it is biased towards
    low values F_X, which we prefer as these are low losses and hence
    good jobs to sample from.

    Args:
        cdf (Array): array of ordered cumulative probability values
        rs (RandomState): random state engine

    Returns:
        idx (int): sampled draw from [0, len(array)]
    """
    cdf = cdf + [1]
    r = rs.rand()
    n = len(cdf)
    for i in range(1, n):
        if cdf[i] >= r:
            return rs.randint(0, i)
    return rs.randint(0, i)


def sample_cdf(cdf_vals, sample_vals, n_samples, rs, cdf, *cdf_args):
    """Sample a cdf with given vals"""
    sampled_cdf = cdf(cdf_vals, *cdf_args)
    samples = [sample_vals[inverse_transformation_sample(sampled_cdf, rs)]
               for _ in range(n_samples)]
    return samples


###############################################################################

class Parameter(object):

    """Parameter class for random sampling

    Class for sampling and perturbing a parameter state.

    Args:
        name (str): name of the parameter
        type (obj): parameter type
        func (tuple): transformation from basic uniform distribution, optional
            Need to supply a tuple of functions for (transform, inverse_transform)
        minmax (tuple): tuple of (min, max) parameter value, optional
        support (list): acceptable parameter states, optional
        seed (int): seed for sampling, perturbation, and re-seeding.
    """

    def __init__(self, name, type, minmax=None, support=None, func=None,
                 perturb_range=None, seed=None, frozen=None):

        if frozen is None:
            if support is not None and minmax is not None:
                raise ValueError("Cannot set both support and minmax")
            if type is not bool and ((support is None) and (minmax is None)):
                raise ValueError("Specify either min-max or support.")
            if type is float and minmax is None:
                raise ValueError("Float variables must have min-max set.")

        self.name = name
        self.type = type
        self.func = func
        self.seed = seed
        self.frozen = frozen
        self.minmax = minmax
        self.support = support

        if perturb_range is None:
            perturb_range = PERTURB_RANGE[type]
        self.perturb_range = perturb_range

    def copy(self):
        """Copy instance

        Returns:
            inst (Parameter): copy of instance
        """
        return Parameter(**self.__dict__)

    def transform(self, value, inverse=False):
        """Set parameter state

        Args:
            value (obj): value to set state to
            inverse (bool): inverse transformation

        Returns:
            transformed (int, float, str, bool): transformed value
        """
        if self.func is not None:
            func = self.func[0] if not inverse else self.func[1]
            value = func(value)
        return value

    def reseed(self, seed=None):
        """Re-seed parameter instance

        Re-seeding uses current seed if no seed is specified to draw a new seed
        from 0-100000000.

        Args:
            seed (int): seed to use to draw new seed, optional

        Returns:
            self (Parameter): re-seeded instance.
        """
        if seed is None:
            seed = self.seed
        rs = RandomState(seed)
        self.seed = rs.randint(0, 100000000)
        return self

    def sample(self, keep_seed=False):
        """Sampling parameter state value

        Args:
            keep_seed (bool): if False, :func:`reseed` will be called
            prior to sampling, optional

        Returns:
              sample (int, float, bool, str): sampled value
        """
        if self.frozen is not None:
            return self.frozen

        if not keep_seed:
            self.reseed()

        return self._sample(self.minmax, self.support)

    def perturb(self, value, keep_seed=False):
        """Perturbed parameters state

        If applicable, parameter will be perturbed according to the
        ``perturb`` attribute.

        Args:
            value (int, float, bool): baseline value to perturb
            keep_seed (bool): if False, :func:`reseed` will be called
            prior to sampling, optional

        Returns:
            perturbed (int, float, bool): sampled perturbation of value
        """
        if self.frozen is not None:
            return self.frozen

        if not self.perturb_range:
            return value

        if not keep_seed:
            self.reseed()

        mx = None
        support = None
        value = self.transform(value, inverse=True)

        if self.perturb_range is True:
            if self.support:
                support = self.support
            if self.minmax:
                mx = self.minmax

        if self.type is float:
            mx = (
                value * (1 - self.perturb_range[0]),
                value * (1 + self.perturb_range[1])
            )

        if self.type is int and self.minmax:
            mx = (
                value - self.perturb_range[0],
                value + self.perturb_range[1]
            )

        if (self.type is int or self.type is str) and self.support:
            # We assume perturbation is wrt index position in this case
            for i, val in enumerate(support):
                if val == value:
                    break
            smin = max(i - self.perturb_range[0], 0)
            smax = min(i + self.perturb_range[1] + 1, len(support))
            support = self.support[smin:smax]

        # Prune outlying support for min-max ranges
        if mx:
            mx = (max(mx[0], self.minmax[0]),
                  min(mx[1], self.minmax[1]))

        return self._sample(mx, support)

    def _sample(self, minmax, support):
        """General sampling process across parameter types"""
        rs = RandomState(seed=self.seed)
        val = None

        if self.type is float:
            val = uniform(*minmax, rs)
        elif self.type is bool:
            # No need to consider minmax or support
            val = rs.rand() >= 0.5
        else:
            if minmax is not None:
                val = rs.randint(*minmax)
            if support is not None:
                val = self.support[rs.randint(0, len(support))]

        return self.transform(val)
