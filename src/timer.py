import time
import logging as log


def timeit(f):
    def timed(*args, **kw):

        ts = time.time()
        result = f(*args, **kw)
        te = time.time()

        logger = log.getLogger()
        logger.info(f"{args[0]} {f.__name__} took {te-ts}")
        return result

    return timed
