import os
import logging

logger = logging.getLogger("gunicorn.error")

# Server socket
bind = f"0.0.0.0:{os.environ.get('PORT', '10000')}"

# Worker processes and thread model
# Exactly 1 worker process with 4 request threads using gthread.
# Memory footprint is ~140-180MB, well below Render Free tier 512MB limit.
workers = 1
worker_class = "gthread"
threads = 4

# Timeouts
timeout = 120
graceful_timeout = 30
keepalive = 5


def post_worker_init(worker):
    """
    Called after a Gunicorn worker process has initialized the Django application.
    Starts the in-process background task worker thread.
    """
    try:
        from curriculum.worker import start_in_process_worker
        thread = start_in_process_worker()
        logger.info(
            "[Gunicorn Hook] In-process generation worker thread [%s] started successfully in worker (pid=%s).",
            thread.name, worker.pid
        )
    except Exception as e:
        logger.exception(
            "[Gunicorn Hook] CRITICAL: Failed to start in-process generation worker in worker (pid=%s): %s",
            worker.pid, e
        )


def worker_exit(server, worker):
    """
    Called in the worker process just after exiting or receiving termination signal.
    Signals the in-process background worker thread to stop and bounds shutdown wait.
    """
    try:
        from curriculum.worker import stop_in_process_worker
        # Bounded shutdown: wait up to 15 seconds (well within graceful_timeout=30s)
        stopped = stop_in_process_worker(timeout=15.0)
        if stopped:
            logger.info(
                "[Gunicorn Hook] In-process generation worker stopped cleanly in worker (pid=%s).",
                worker.pid
            )
        else:
            logger.warning(
                "[Gunicorn Hook] In-process generation worker in worker (pid=%s) timed out during shutdown.",
                worker.pid
            )
    except Exception as e:
        logger.warning(
            "[Gunicorn Hook] Error during in-process generation worker shutdown in worker (pid=%s): %s",
            worker.pid, e
        )
