import functools
import logging
import os
from contextlib import contextmanager
from pathlib import Path

from joblib import parallel
from rich.progress import track

logger = logging.getLogger(__name__)


@contextmanager
def track_joblib(*args, **kwargs):
    """
    Context manager to patch joblib to report 
    into rich progress bar given as argument
    """
    track_object = track(*args, **kwargs)

    class TrackBatchCompletionCallback(parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            track_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = parallel.BatchCompletionCallBack
    parallel.BatchCompletionCallBack = TrackBatchCompletionCallback
    try:
        yield track_object
    finally:
        parallel.BatchCompletionCallBack = old_batch_callback
        track_object.close()


def catch_obsid_error(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        try:
            value = func(*args, **kwargs)
        
        except Exception as e:
            logger.error(e, exc_info=1)
            logger.error(f"Error processing Obs.ID. {args[0]:010}")
            value = None

        return value
    
    return wrapper_decorator


@contextmanager
def working_directory(path):
    """Changes working directory and returns to previous on exit."""
    prev_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


@contextmanager
def all_logging_disabled(highest_level=logging.CRITICAL):
    """
    A context manager that will prevent any logging messages
    triggered during the body from being processed.
    :param highest_level: the maximum logging level in use.
      This would only need to be changed if a custom level greater than CRITICAL
      is defined.
    """
    # two kind-of hacks here:
    #    * can't get the highest logging level in effect => delegate to the user
    #    * can't get the current module-level override => use an undocumented
    #       (but non-private!) interface
    # https://gist.github.com/simon-weber/7853144

    previous_level = logging.root.manager.disable

    logging.disable(highest_level)

    try:
        yield
    finally:
        logging.disable(previous_level)
