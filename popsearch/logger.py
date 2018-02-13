"""
Customized logger: use as a normal logger.
"""
import os
import time
import logging
from .eval import eval_step

class Logger(logging.getLoggerClass()):

    """A Logger class with pre-set file formatting

    Args:
        logger_count (int): the logger id
        log_path (str): the parent directory of log files
        **params: parameters of job configuration
    """

    def __init__(self, jid, path, **params):
        jid = int(jid)
        path = str(path)
        self.jid = jid
        self.path = path
        self.params = params

        # Set path
        date = time.gmtime()
        _date = '%s-%s-%s-%s' % (
            str(date.tm_year),
            str(date.tm_mon),
            str(date.tm_mday),
            str(date.tm_hour)
        )
        name = '{}:'.format(jid) + _date
        path = os.path.join(path, name + '.log')
        if os.path.exists(path):
            os.unlink(path)

        # Initialize logger
        super(Logger, self).__init__(name)

        self.setLevel(logging.DEBUG)
        formatting = logging.Formatter('%(message)s')
        fh = logging.FileHandler(path)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatting)
        self.addHandler(fh)

        pars = ""
        for i, (k, v) in enumerate(self.params.items()):
            if i == 0:
                pars += "{}-{}-{}".format(k, type(v), v)
            else:
                pars += ",{}-{}-{}".format(k, type(v), v)

        self.info("JOB:{}\nDATE:{}\nPARAMS:{}".format(jid, _date, pars))

    def eval(self, step, value):
        self.info("EVAL:{}:{}".format(step, value))
        return eval_step(self.path, value, step, self.jid)
