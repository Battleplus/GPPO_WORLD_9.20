"""Task accounting for the new M-10 environment, not the frozen GPPO baseline.

Assignment is a reservation, not task completion. Time units and service rates
are supplied by the environment; this module does not model flight physics.
"""
from dataclasses import dataclass
from enum import Enum
import math


class TaskState(str, Enum):
    UNRELEASED = "unreleased"
    PENDING = "pending"
    ASSIGNED = "assigned"
    SERVING = "serving"
    COMPLETED = "completed"
    EXPIRED = "expired"


@dataclass
class TaskLifecycle:
    task_id: str
    arrival: float
    deadline: float
    required_service: float
    priority: float = 1.0
    state: TaskState = TaskState.UNRELEASED
    assigned_uav: str | None = None
    service: float = 0.0
    completed_at: float | None = None
    last_time: float = -math.inf

    def __post_init__(self):
        if not all(math.isfinite(x) for x in
                   (self.arrival, self.deadline, self.required_service, self.priority)):
            raise ValueError("Task parameters must be finite")
        if self.arrival < 0 or self.deadline <= self.arrival:
            raise ValueError("Invalid release/deadline interval")
        if self.required_service <= 0 or self.priority <= 0:
            raise ValueError("Service and priority must be positive")
        if (self.state != TaskState.UNRELEASED or self.assigned_uav is not None
                or self.service != 0 or self.completed_at is not None
                or self.last_time != -math.inf):
            raise ValueError("Construct tasks in their initial state")

    def advance(self, now: float):
        if not math.isfinite(now) or now < 0 or now < self.last_time:
            raise ValueError("Time must be finite and monotonic")
        self.last_time = now
        if self.state in (TaskState.COMPLETED, TaskState.EXPIRED):
            return
        if now >= self.deadline:
            self.state = TaskState.EXPIRED
            self.assigned_uav = None
        elif now >= self.arrival and self.state == TaskState.UNRELEASED:
            self.state = TaskState.PENDING

    def assign(self, uav: str, now: float):
        if not uav:
            raise ValueError("UAV identity required")
        self.advance(now)
        if self.state != TaskState.PENDING:
            raise ValueError("Only pending tasks can be assigned")
        self.assigned_uav = uav
        self.state = TaskState.ASSIGNED

    def interrupt(self, now: float):
        self.advance(now)
        if self.state in (TaskState.ASSIGNED, TaskState.SERVING):
            self.assigned_uav = None
            self.state = TaskState.PENDING

    def provide_service(self, uav: str, start: float, end: float, rate: float):
        """Environment must certify uninterrupted, energy-feasible service.

        Split intervals at failures/communication transitions before calling.
        Work accumulated before an interruption is retained by this contract.
        Completion exactly at the deadline is allowed.
        """
        if not all(math.isfinite(x) for x in (start, end, rate)):
            raise ValueError("Service interval must be finite")
        if end <= start or rate <= 0:
            raise ValueError("Positive duration and rate required")
        self.advance(start)
        if self.assigned_uav != uav or self.state not in (TaskState.ASSIGNED, TaskState.SERVING):
            raise ValueError("Service requires the assigned UAV")
        finish = start + (self.required_service - self.service) / rate
        if finish <= min(end, self.deadline):
            self.service = self.required_service
            self.completed_at = finish
            self.state = TaskState.COMPLETED
            self.assigned_uav = None
        else:
            self.service += (min(end, self.deadline) - start) * rate
            self.state = TaskState.SERVING
        self.advance(end)
