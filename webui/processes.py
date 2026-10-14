"""Stream and stop only subprocesses started by this application."""

import os
import signal
import subprocess
import threading
from collections import deque
from dataclasses import dataclass, field
from queue import Empty, Queue


@dataclass
class ProcessEvent:
    log: str
    returncode: int | None = None
    cancelled: bool = False


@dataclass
class _Job:
    process: subprocess.Popen
    cancelled: threading.Event = field(default_factory=threading.Event)


def _terminate(process):
    if os.name != "posix" and process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait()
    except ProcessLookupError:
        pass
    finally:
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


class ProcessManager:
    def __init__(self):
        self._jobs: dict[str, _Job] = {}
        self._lock = threading.Lock()

    def stream(self, job_id, command, *, cwd=None, gpu="", secrets=()):
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        if str(gpu).strip():
            env["CUDA_VISIBLE_DEVICES"] = str(gpu).strip()
        with self._lock:
            if job_id in self._jobs:
                raise ValueError("该任务正在运行，请先停止或等待完成。")
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=env,
                shell=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                start_new_session=os.name == "posix",
            )
            job = _Job(process)
            self._jobs[job_id] = job
        queue = Queue(maxsize=1000)
        closed = threading.Event()

        def read_output():
            from queue import Full

            def put(value):
                while not closed.is_set():
                    try:
                        queue.put(value, timeout=0.1)
                        return
                    except Full:
                        continue

            try:
                for line in process.stdout:
                    put(line.rstrip("\n"))
                    if closed.is_set():
                        break
            finally:
                put(None)

        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        lines = deque(maxlen=1000)
        try:
            yield ProcessEvent("已启动，等待后端输出…")
            while True:
                try:
                    line = queue.get(timeout=0.2)
                except Empty:
                    continue
                if line is None:
                    break
                for secret in secrets:
                    if secret:
                        line = line.replace(str(secret), "[redacted]")
                lines.append(line[-20_000:])
                yield ProcessEvent("\n".join(lines)[-100_000:])
            code = process.wait()
            yield ProcessEvent("\n".join(lines)[-100_000:], code, job.cancelled.is_set())
        finally:
            closed.set()
            _terminate(process)
            reader.join(timeout=2)
            process.stdout.close()
            with self._lock:
                self._jobs.pop(job_id, None)

    def stop(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False
            job.cancelled.set()
        _terminate(job.process)
        return True

    def close(self):
        with self._lock:
            job_ids = list(self._jobs)
        for job_id in job_ids:
            self.stop(job_id)


processes = ProcessManager()
