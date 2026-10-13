import os, time
from contextlib import contextmanager

LOG = os.path.abspath(os.path.join(os.path.dirname(__file__),"..", "..","data","output" "timing.log"))

def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")

@contextmanager
def timer(label):
    t = time.perf_counter()
    try:
        yield
    finally:
        log(f"[timing] {label}: {time.perf_counter() - t:.2f}s")