"""In-memory registry for background batch jobs."""
import threading
import time
import uuid


class Job:
    def __init__(self, total: int, spec: dict):
        self.id = uuid.uuid4().hex[:12]
        self.total = total
        self.done = 0
        self.status = "running"   # running | complete | error
        self.spec = spec
        self.result = None
        self.error = None
        self.current = None       # {case_id, voice, steps, rep}
        self.started = time.perf_counter()

    def snapshot(self) -> dict:
        return {
            "job_id": self.id,
            "status": self.status,
            "done": self.done,
            "total": self.total,
            "current": self.current,
            "elapsed_s": round(time.perf_counter() - self.started, 1),
            "error": self.error,
        }


class JobRegistry:
    def __init__(self):
        self._jobs = {}
        self._lock = threading.Lock()

    def create(self, total: int, spec: dict) -> Job:
        job = Job(total, spec)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str):
        with self._lock:
            return self._jobs.get(job_id)


registry = JobRegistry()
