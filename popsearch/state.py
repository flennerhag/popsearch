"""Job state class.
"""
import os
import time
import logging
from numpy.random import RandomState


class State(object):

    """Job state class

    Class containing job data.

    Args:
        jid (int): job number id
        config (Config): a :class:`Config` instance with job meta data
        parameters (dict): dictionary of parameter values for the job
        force (bool): force complete
    """

    def __init__(self, jid, config, parameters, force=None):
        self.step = 0
        self.jid = jid
        self.config = config
        self.parameters = parameters
        self.force = force
        self.path = config.path
        self.max_val = config.max_val
        self.tolerance = config.tolerance
        self.min_eval_step = config.min_eval_step
        self._eval, *self._eval_args = config.eval_rule

        self.values = []
        self.logger = None
        self.status = None
        self.rs = RandomState(self.jid)

    def __bool__(self):
        """Evaluate state"""
        return self.status == 1

    def __nonzero__(self):
        """Evaluate state (2.7 comp)"""
        return self.status == 1

    def eval(self, value):
        """Evaluate job at checkpoint step"""
        self.step += 1
        self.values.append(value)

        status = None
        if self.force:
            if self.step <= self.config.force_n_step:
                status = 1
        if self.max_val is not None:
            if value > self.max_val:
                status = 0
        if self.tolerance is not None and len(self.values) >= self.tolerance:
            if self.values[-self.tolerance] < value:
                status = 0

        if status is None:
            status = self._eval(self, *self._eval_args)

        self.status = status

        if self.logger is None:
            self.logger = Logger(self)
        self.info("EVAL:{}:{}".format(self.step, value))

        if self.status == 0:
            self.logger = None

    def info(self, msg, *args, **kwargs):
        """Record job information"""
        if self.logger is None:
            self.logger = Logger(self)
        self.logger.info(msg, *args, **kwargs)


class Logger(logging.getLoggerClass()):

    """A Logger class with pre-set file formatting

    Args:
        state (State): the :class:`State` instance to create the logger for.
    """

    def __init__(self, state):
        self.jid = state.jid
        self.path = state.path
        self.params = state.parameters

        # Set path
        gmtime = time.gmtime()
        date = '%s-%s-%s-%s' % (
            str(gmtime.tm_year),
            str(gmtime.tm_mon),
            str(gmtime.tm_mday),
            str(gmtime.tm_hour),
        )

        path = os.path.join(self.path, str(self.jid) + '.log')

        # Initialize logger
        super(Logger, self).__init__(str(self.jid))

        self.setLevel(logging.INFO)
        formatting = logging.Formatter('%(message)s')

        fh = logging.FileHandler(path)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatting)
        self.addHandler(fh)

        pars = ""
        for i, (k, v) in enumerate(self.params.items()):
            t = str(type(v)).split("'")[1]
            if i != 0:
                pars += ";"
            pars += "{},{},{}".format(k, t, v)

        self.info("JOB:{}\nDATE:{}\nPARAMS:{}".format(self.jid, date, pars))
