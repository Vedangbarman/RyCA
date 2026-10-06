import itertools, queue, threading
from concurrent.futures import Future

NOTIFY, CHAT = 0, 1                 # lower number goes first
_jobs = queue.PriorityQueue()
_count = itertools.count()          # keeps same-priority jobs in order

def _worker():
    while True:
        _, _, fn, args, fut = _jobs.get()
        try:
            fut.set_result(fn(*args))
        except Exception as e:
            fut.set_exception(e)

threading.Thread(target=_worker, daemon=True).start()

def submit(priority, fn, *args):
    fut = Future()
    _jobs.put((priority, next(_count), fn, args, fut))
    return fut