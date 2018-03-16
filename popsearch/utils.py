"""Utility functions
"""
import os


def get_logs(p):
    """Get log files from path"""
    return [os.path.join(p, l) for l in os.listdir(p) if l.endswith('.log')]


def get_jid(log_name):
    """Strip path and file type from the log name to return the jid as int"""
    return int(log_name.split('/')[-1].split('.')[0])


def jid_to_log(jids, path="", logs=None):
    """Construct log files from a list of jids, or larger set of logs"""
    if not logs:
        return [os.path.join(path, "{}.log".format(jid)) for jid in jids]
    return [l for l in logs if get_jid(l) in jids]
