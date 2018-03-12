"""Job state class.
"""
import os
import time
import logging
from .eval import eval_step


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
        self.status = None
        self.config = config
        self.parameters = parameters
        self.force = force
        self.path = config.path
        self.max_val = config.max_val
        self.logger = None

    def __bool__(self):
        """Evaluate state"""
        return self.status == 1

    def __nonzero__(self):
        """Evaluate state (2.7 comp)"""
        return self.status == 1

    def eval(self, value):
        """Evaluate job at checkpoint step"""
        self.step += 1

        status = None
        if self.force:
            status = 1
        if self.max_val is not None and value > self.max_val:
            status = 0
        if status is None:
            status = eval_step(self.path, value, self.step)

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
        with open(path, 'r') as f:
            lines = f.read()
            if lines != "":
                raise OSError("Job file already exists")

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
