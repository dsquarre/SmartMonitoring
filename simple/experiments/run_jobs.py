"""Run a job file: one `run.py` argument line per job (blank lines and # comments ignored), at most --workers at a time.
Finished runs are skipped by run.py itself, so the same job file can be re-run after an interruption.

    python experiments/run_jobs.py experiments/jobs_calibration.txt --workers 2
    python experiments/run_jobs.py experiments/jobs_tests.txt --log-every 5     # quieter console

Live progress: every round of every running job is printed as it finishes, prefixed with the job number and followed
by the overall progress (rounds done / total) and an ETA from the speed of this session. --log-every N prints every
N-th round instead. The full output of each job (including TensorFlow messages) goes to <jobs>_logs/job_NNN.log.
Job lines are split like a POSIX shell: write paths with forward slashes (a backslash is dropped).
"""
import argparse
import os
import re
import shlex
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
SIMPLE = os.path.dirname(HERE)
DEFAULT_ROUNDS = 100                     # SimConfig.rounds
ROUND_RE = re.compile(r"^\[(?P<label>\S+) s(?P<seed>\d+)\] round\s+(?P<t>\d+) ")
RESUME_RE = re.compile(r"^\[\S+ s\d+\] resuming after round (?P<t>\d+)")


def job_size(line):
    """(seed-runs, rounds per seed-run) of one job line, parsed like run.py does."""
    args = shlex.split(line)

    def value(flag, default):
        return args[args.index(flag) + 1] if flag in args else default
    seeds = 0
    for part in value("--seeds", "42").split(","):
        a, _, b = part.partition("-")
        seeds += int(b) - int(a) + 1 if b else 1
    strategies = len([s for s in value("--strategy", "select_train").split(",") if s.strip()])
    return seeds * strategies, int(value("--rounds", DEFAULT_ROUNDS))


class Progress:
    def __init__(self, lines):
        self.lock = threading.Lock()
        self.sizes = [job_size(l) for l in lines]
        self.total = sum(n * r for n, r in self.sizes)
        self.done_runs = [0] * len(lines)        # finished seed-runs per job
        self.cur_round = [0] * len(lines)        # round reached in the running seed-run
        self.new_rounds = 0                      # rounds trained in this session (for the ETA)
        self.t0 = time.time()

    def done(self):
        return sum(d * r + c for d, (_, r), c in zip(self.done_runs, self.sizes, self.cur_round))

    def status(self):
        done = self.done()
        pct = 100.0 * done / max(self.total, 1)
        if self.new_rounds:
            eta = (time.time() - self.t0) / self.new_rounds * (self.total - done)
            eta_s = f"ETA {int(eta // 3600)}h{int(eta % 3600 // 60):02d}m"
        else:
            eta_s = "ETA --"
        return f"{done}/{self.total} rounds ({pct:.1f}%), {eta_s}"

    def say(self, msg):
        with self.lock:
            print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_one(i, n, line, log_dir, prog, log_every):
    t0 = time.time()
    log = os.path.join(log_dir, f"job_{i:03d}.log")
    k = i - 1
    prog.say(f"job {i}/{n} start: {line}")
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    with open(log, "w") as f:
        proc = subprocess.Popen([sys.executable, "run.py", "--log-every", "1"] + shlex.split(line), cwd=SIMPLE,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
        for out in proc.stdout:
            f.write(out)
            f.flush()
            out = out.rstrip()
            m = ROUND_RE.match(out)
            if m:
                t = int(m.group("t"))
                with prog.lock:
                    prog.cur_round[k] = t
                    prog.new_rounds += 1
                if t % log_every == 0 or t == prog.sizes[k][1]:
                    prog.say(f"job {i}/{n} {out} | {prog.status()}")
            elif RESUME_RE.match(out):
                with prog.lock:
                    prog.cur_round[k] = int(RESUME_RE.match(out).group("t"))
                prog.say(f"job {i}/{n} {out}")
            elif out.endswith(".csv"):                    # a seed-run finished (or was already done)
                with prog.lock:
                    prog.done_runs[k] += 1
                    prog.cur_round[k] = 0
            elif "diverged" in out or "Traceback" in out or "Error" in out:
                prog.say(f"job {i}/{n} {out}")
        rc = proc.wait()
    if rc == 0:                                           # count anything not seen (e.g. skipped) as done
        with prog.lock:
            prog.done_runs[k], prog.cur_round[k] = prog.sizes[k][0], 0
    status = "ok" if rc == 0 else f"FAILED (rc={rc}, see {log})"
    prog.say(f"job {i}/{n} {status} in {(time.time() - t0) / 60:.1f} min | {prog.status()}: {line}")
    return rc


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("jobs")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--log-every", type=int, default=1, help="print every N-th round of each run (default: every)")
    a = p.parse_args()
    with open(a.jobs) as f:
        lines = [l.strip() for l in f if l.strip() and not l.strip().startswith("#")]
    log_dir = os.path.splitext(os.path.abspath(a.jobs))[0] + "_logs"
    os.makedirs(log_dir, exist_ok=True)
    prog = Progress(lines)
    print(f"{len(lines)} jobs ({prog.total} rounds), {a.workers} at a time, logs in {log_dir}", flush=True)
    with ThreadPoolExecutor(a.workers) as pool:
        codes = list(pool.map(lambda x: run_one(x[0] + 1, len(lines), x[1], log_dir, prog, max(a.log_every, 1)),
                              enumerate(lines)))
    failed = sum(c != 0 for c in codes)
    print(f"done: {len(codes) - failed} ok, {failed} failed in {(time.time() - prog.t0) / 60:.1f} min", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
