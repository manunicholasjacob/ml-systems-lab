"""Run a sweep across several machines at once, and survive one of them dying.

The sequential runner in :mod:`mlsyslab.runner` is still the reference implementation and
is unchanged. This module drives the same :class:`~mlsyslab.runner.Runner` from several
threads, which buys wall-clock time on a heterogeneous set of machines where the slowest
device would otherwise set the pace for all of them.

Four things here are load-bearing, and each of them exists because the naive version is
wrong in a way that quietly damages the data rather than loudly failing.

**Concurrency is per device, not global.** A thread pool with one number cannot express
"the laptop can take four of these and the Pi can take exactly one". The 2 GB Pi has hung
hard enough to need a physical power cycle when given two heavy jobs at once, so its limit
is a property of the device and it defaults to 1 for everything.

**Some devices are the same machine.** A host and the pods scheduled onto it are four
device ids and one CPU, and running them at once would not be a faster sweep, it would be
four measurements of contention. Devices that share hardware declare a ``resource_group``,
which holds one job at a time across all of its members unless told otherwise. This is the
difference between a scheduler for a compute cluster and a scheduler for an experiment.

**Only transport failures are retried.** A dropped SSH connection is worth another try. A
model that will not fit in RAM is a *result*, and retrying it three times would turn a
finding into a hole in the matrix with nothing to show for it. :class:`Attempt` carries
that distinction and this module never second-guesses it.

**Every retry is written down.** A benchmark that silently retries is a benchmark you
cannot trust, because the reader has no way to know whether a number came from a clean
run or from the third attempt on a machine that was already misbehaving. Retries land in
``retries.jsonl`` and in the record's own warnings.

**Work is not moved between devices by default.** In most schedulers, requeuing a task
elsewhere is the obvious response to a dead worker. In a hardware benchmark it is close
to the worst thing you can do, because the device *is* the independent variable: a Pi
measurement taken on the laptop is not a repaired data point, it is a different
experiment wearing the same run id. Failover therefore happens only where a config
explicitly declares two devices interchangeable, and the moved run is tagged so that no
analysis can mistake it for the original.
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .backends.base import RunSpec
from .runner import Attempt, Runner

# Deliberately conservative. A device that has not said otherwise gets one job at a time,
# because the cost of guessing high is a corrupted measurement or a hung board, and the
# cost of guessing low is a slower sweep.
DEFAULT_MAX_CONCURRENCY = 1


@dataclass
class RetryPolicy:
    """Bounded, backed off, and written down. All three matter."""

    max_attempts: int = 3            # total tries per spec, not retries after the first
    backoff_s: float = 5.0
    backoff_factor: float = 2.0
    max_backoff_s: float = 120.0
    jitter: float = 0.25             # fraction of the delay, so devices do not resynchronise

    def delay_for(self, attempt: int, rng: Optional[random.Random] = None) -> float:
        """Delay before attempt number ``attempt`` (2 for the first retry)."""
        base = min(self.backoff_s * (self.backoff_factor ** max(0, attempt - 2)),
                   self.max_backoff_s)
        spread = base * self.jitter
        draw = (rng or random).uniform(-spread, spread)
        return max(0.0, base + draw)


@dataclass
class DevicePolicy:
    """What one device is allowed to be asked to do, and when to stop asking."""

    device_id: str
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY
    # Consecutive transport failures before the breaker opens. Consecutive, not total:
    # a device that fails once an hour over a long sweep is annoying, not dead.
    max_transport_failures: int = 3
    recovery_after_s: float = 180.0
    failover_to: List[str] = field(default_factory=list)
    # Devices sharing a name here share physical hardware, and the group runs
    # ``group_max_concurrency`` jobs at a time across all of them. One by default,
    # because the only reason to declare a group is that these devices interfere.
    resource_group: Optional[str] = None
    group_max_concurrency: int = 1

    @classmethod
    def from_config(cls, device_id: str, config: Dict[str, Any]) -> "DevicePolicy":
        config = config or {}
        failover = config.get("failover_to") or []
        if isinstance(failover, str):
            failover = [failover]
        return cls(
            device_id=device_id,
            max_concurrency=max(1, int(config.get("max_concurrency")
                                       or DEFAULT_MAX_CONCURRENCY)),
            max_transport_failures=max(1, int(config.get("max_transport_failures") or 3)),
            recovery_after_s=float(config.get("recovery_after_s") or 180.0),
            failover_to=list(failover),
            resource_group=config.get("resource_group") or None,
            group_max_concurrency=max(1, int(config.get("group_max_concurrency") or 1)),
        )

    @property
    def group(self) -> str:
        """The pool this device draws from. Its own name when it shares with nobody."""
        return self.resource_group or f"__device__{self.device_id}"


class _Breaker:
    """A circuit breaker per device: closed, open, then half open for one probe.

    Without this, one unreachable machine costs the sweep its full retry budget on every
    remaining point on that machine, which on a 60 point matrix is an hour of sleeping.
    With it, the device is taken out of service after a few consecutive failures and
    given exactly one chance to come back later. The half-open probe is what makes a
    tailnet that drops for two minutes a delay rather than a lost night.
    """

    def __init__(self, policy: DevicePolicy):
        self.policy = policy
        self.consecutive_failures = 0
        self.opened_at: Optional[float] = None
        # opened_at restarts on every failed probe, which is what the cooldown wants and
        # exactly what a "has this device been down too long" test must not use: it would
        # never fire. This one is set when the device first goes down and cleared only by
        # a run that actually worked.
        self.down_since: Optional[float] = None
        self.reason: Optional[str] = None
        self.probe_in_flight = False
        self.trips = 0
        self.recoveries = 0

    @property
    def is_open(self) -> bool:
        return self.opened_at is not None

    def down_for(self, now: float) -> float:
        """Seconds since the device first stopped answering and never came back."""
        return 0.0 if self.down_since is None else max(0.0, now - self.down_since)

    def record_success(self) -> bool:
        """Returns True if this closed an open breaker."""
        self.consecutive_failures = 0
        self.probe_in_flight = False
        self.down_since = None
        if self.opened_at is not None:
            self.opened_at = None
            self.reason = None
            self.recoveries += 1
            return True
        return False

    def record_failure(self, reason: str, now: float) -> bool:
        """Returns True if this opened the breaker."""
        self.consecutive_failures += 1
        if self.probe_in_flight:
            # The half-open probe failed. Re-open and start the cooldown again, but leave
            # down_since where it was: the device has been down since the first trip.
            self.probe_in_flight = False
            self.opened_at = now
            self.reason = reason
            return False
        if self.opened_at is None and \
                self.consecutive_failures >= self.policy.max_transport_failures:
            self.opened_at = now
            self.down_since = self.down_since or now
            self.reason = reason
            self.trips += 1
            return True
        return False

    def would_allow(self, now: float) -> bool:
        """The same question as :meth:`allow`, asked without consuming the probe.

        Needed by the fairness pass, which has to look at several devices before it
        decides which one goes next. Asking with ``allow`` would burn a half-open
        probe on every device it merely considered.
        """
        if self.opened_at is None:
            return True
        if self.probe_in_flight:
            return False
        return now - self.opened_at >= self.policy.recovery_after_s

    def allow(self, now: float) -> bool:
        """May a worker take another job on this device right now?"""
        if self.opened_at is None:
            return True
        if self.probe_in_flight:
            return False
        if now - self.opened_at >= self.policy.recovery_after_s:
            self.probe_in_flight = True   # exactly one job goes through to test the water
            return True
        return False


@dataclass
class SweepResult:
    records: List[Any] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    retries: List[Dict[str, Any]] = field(default_factory=list)
    clock_offsets: Dict[str, Any] = field(default_factory=dict)


class ConcurrentSweep:
    """Schedule specs across devices, with per-device limits and partial failure."""

    def __init__(
        self,
        runner: Runner,
        policies: Optional[Dict[str, DevicePolicy]] = None,
        retry: Optional[RetryPolicy] = None,
        on_event: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        max_workers: int = 32,
        rng_seed: Optional[int] = None,
    ):
        self.runner = runner
        self.retry = retry or RetryPolicy()
        self.on_event = on_event or runner.on_event
        self.max_workers = max_workers
        self._rng = random.Random(rng_seed)

        self.policies: Dict[str, DevicePolicy] = dict(policies or {})
        for device_id, device_config in runner.config.devices.items():
            self.policies.setdefault(
                device_id, DevicePolicy.from_config(device_id, device_config)
            )

        self._queues: Dict[str, List[Tuple[RunSpec, int]]] = {}
        self._breakers: Dict[str, _Breaker] = {}
        # One condition guards the queues, the breakers and the in-flight count. Workers
        # wait on it rather than polling, and, more importantly, a worker may only exit
        # when nothing is queued anywhere *and* nobody is still running: a failover can
        # put work on a device whose own queue drained minutes ago, and a worker that
        # had already returned would leave that work stranded.
        self._cond = threading.Condition()
        self._inflight = 0
        self._group_inflight: Dict[str, int] = {}
        self._group_order: Dict[str, List[str]] = {}
        self._stop = threading.Event()
        self._records: List[Any] = []
        self._retry_log: List[Dict[str, Any]] = []
        self._done = 0
        self._total = 0
        self._skipped = 0

    # ------------------------------------------------------------------ queue state

    def _policy(self, device_id: str) -> DevicePolicy:
        if device_id not in self.policies:
            self.policies[device_id] = DevicePolicy.from_config(
                device_id, self.runner.config.devices.get(device_id, {})
            )
        return self.policies[device_id]

    def _staffing(self, seeds: set) -> set:
        """Devices that need a worker: the ones with work, plus their failover closure."""
        known = set(self.runner.config.devices) or set(seeds)
        staffed, frontier = set(seeds), list(seeds)
        while frontier:
            device_id = frontier.pop()
            for candidate in self._policy(device_id).failover_to:
                if candidate not in staffed and candidate in known:
                    staffed.add(candidate)
                    frontier.append(candidate)
        return staffed

    def _breaker(self, device_id: str) -> _Breaker:
        if device_id not in self._breakers:
            self._breakers[device_id] = _Breaker(self._policy(device_id))
        return self._breakers[device_id]

    def _group_limit(self, group: str) -> int:
        """How many jobs the shared hardware behind this group will take at once.

        The tightest limit any member declares wins. A device that thinks its box can
        only do one thing at a time is not overruled by a sibling that disagrees.
        """
        members = [p for p in self.policies.values() if p.group == group]
        if not members:
            return 1
        if group.startswith("__device__"):
            # A device that shares with nobody is its own group, and its ceiling is
            # simply its own concurrency. Anything else here would quietly override
            # max_concurrency, which is the one number a device owner actually sets.
            return members[0].max_concurrency
        return min(p.group_max_concurrency for p in members)

    def _group_members(self, group: str) -> List[str]:
        return sorted(d for d, p in self.policies.items() if p.group == group)

    def _group_has_room(self, group: str) -> bool:
        return self._group_inflight.get(group, 0) < self._group_limit(group)

    def _is_this_devices_turn(self, device_id: str, group: str) -> bool:
        """Round robin across the devices sharing one piece of hardware.

        Without this the group is first come first served, and first come is whichever
        worker happens to reacquire the lock after releasing it, which is reliably the
        one that just finished. In a five arm study on one CPU that meant one arm ran
        its entire queue before any other arm started, so every arm sat in a different
        part of the machine's thermal history. The experiment would have measured the
        order of the sweep as much as the thing it was supposed to measure.

        Devices with nothing queued, and devices whose breaker is holding them back, do
        not hold up the rotation.
        """
        members = self._group_members(group)
        if len(members) <= 1:
            return True
        now = time.time()
        waiting = [d for d in members
                   if self._queues.get(d) and self._breaker(d).would_allow(now)]
        if not waiting:
            return True
        order = self._rotation(group, members)
        for candidate in order:
            if candidate in waiting:
                return candidate == device_id
        return True

    def _rotation(self, group: str, members: List[str]) -> List[str]:
        """The turn order, kept in step with the group's membership.

        A device can join a group after the rotation was first built, when a failover
        target is staffed. A member missing from the order would never come up for its
        turn and would wait forever, so anything new joins at the back.
        """
        order = self._group_order.setdefault(group, list(members))
        for member in members:
            if member not in order:
                order.append(member)
        return order

    def _rotate_group(self, group: str, device_id: str) -> None:
        order = self._rotation(group, self._group_members(group))
        if device_id in order:
            order.remove(device_id)
            order.append(device_id)

    def _take_locked(self, device_id: str) -> Optional[Tuple[RunSpec, int]]:
        """Pop the next job for a device. Caller holds the condition."""
        queue = self._queues.get(device_id) or []
        if not queue:
            return None
        group = self._policy(device_id).group
        if not self._group_has_room(group):
            return None
        if not self._is_this_devices_turn(device_id, group):
            return None
        if not self._breaker(device_id).allow(time.time()):
            return None
        self._rotate_group(group, device_id)
        return queue.pop(0)

    def _requeue(self, device_id: str, spec: RunSpec, attempts: int) -> None:
        with self._cond:
            self._queues.setdefault(device_id, []).append((spec, attempts))
            self._cond.notify_all()

    def _pending_locked(self) -> int:
        return sum(len(q) for q in self._queues.values())

    # --------------------------------------------------------------------- clocks

    def check_clocks(self, device_ids: List[str]) -> Dict[str, Any]:
        """Compare every device's wall clock against this host's, before measuring.

        Cross-device timing that is compared after the fact is only as good as the worst
        clock in the set, and a board whose NTP never came up can be minutes out without
        anything looking wrong. The offset is recorded whether or not it is alarming,
        because the reader of a results directory should not have to take it on trust.

        The estimate is Cristian's algorithm: the midpoint of the agent's own execution
        minus the midpoint of the round trip. Using the agent's *midpoint* rather than
        either end is what makes it unbiased. An earlier version used only the end
        stamp, and reported four devices on one physical machine as being two to three
        seconds apart, which is nonsense: they share a clock. The bias was the agent's
        own runtime, which is a couple of seconds because collecting sysinfo shells out
        several times. The residual bias is now half the difference between the outbound
        and return legs, which is small for any symmetric transport.

        The uncertainty reported alongside is the transport time that is not accounted
        for, plus the resolution of the stamps. A device whose offset is inside its own
        uncertainty is reported as agreeing, not as slightly wrong.
        """
        offsets: Dict[str, Any] = {}
        for device_id in device_ids:
            entry: Dict[str, Any] = {}
            try:
                device = self.runner.device(device_id)
                t0 = time.time()
                result = device.execute({"kind": "sysinfo"}, timeout_s=180)
                t1 = time.time()
                finished = result.get("agent_epoch_s")
                # `.get(key, default)` is not enough: an older agent can send the key
                # with a null value, and None would then reach the arithmetic below.
                started = result.get("agent_started_epoch_s")
                if started is None:
                    started = finished
                if finished is None:
                    stamp = result.get("agent_time_utc")
                    finished = started = _parse_utc(stamp) if stamp else None
                if finished is None:
                    entry = {"status": "unknown", "note": "agent reported no clock"}
                else:
                    rtt = t1 - t0
                    agent_time = max(0.0, float(finished) - float(started))
                    remote_midpoint = (float(started) + float(finished)) / 2.0
                    offset = remote_midpoint - (t0 + rtt / 2.0)
                    # Only the transport is unaccounted for; the agent's own runtime is
                    # measured and cancels out of the midpoint.
                    uncertainty = max(0.0, (rtt - agent_time)) / 2.0 + 0.002
                    entry = {
                        "status": "ok",
                        "offset_s": round(offset, 3),
                        "round_trip_s": round(rtt, 3),
                        "agent_runtime_s": round(agent_time, 3),
                        # Anything inside this band is indistinguishable from agreement.
                        "uncertainty_s": round(uncertainty, 3),
                    }
                    if abs(offset) > max(1.0, uncertainty):
                        entry["warning"] = (
                            f"clock differs from the host by {offset:+.1f} s, which is "
                            f"outside the {uncertainty:.3f} s this measurement can "
                            "explain; cross-device timing comparisons are not safe "
                            "until NTP is fixed"
                        )
            except Exception as exc:
                entry = {"status": "unreachable", "error": f"{type(exc).__name__}: {exc}"}
            offsets[device_id] = entry
            self.on_event("clock", {"device": device_id, **entry})
        return offsets

    # ------------------------------------------------------------------- execution

    def run(self, specs: Optional[List[RunSpec]] = None,
            check_clocks: bool = True) -> SweepResult:
        specs = list(specs if specs is not None else self.runner.config.specs)
        os.makedirs(self.runner.output_dir, exist_ok=True)
        self.runner._write_manifest(specs)

        pending: List[RunSpec] = []
        for spec in specs:
            if self.runner.resume and self.runner.already_done(spec):
                self._skipped += 1
                self.on_event("skipped", {"spec": spec, "index": 0, "total": len(specs)})
            else:
                pending.append(spec)

        self._total = len(pending)
        for spec in pending:
            self._queues.setdefault(spec.device_id, []).append((spec, 0))

        # Staff every device that could end up holding work, not just the ones the
        # matrix names. A failover target whose own queue starts empty still needs a
        # worker, or the moved point sits in a queue nobody is reading and the sweep
        # never finishes. Found by a test; it would have been a hang at 3am otherwise.
        device_ids = sorted(self._staffing(set(self._queues)))
        offsets = self.check_clocks(sorted(self._queues)) \
            if (check_clocks and self._queues) else {}

        started = time.time()
        threads: List[threading.Thread] = []
        budget = self.max_workers
        for device_id in device_ids:
            slots = min(self._policy(device_id).max_concurrency, max(1, budget))
            budget -= slots
            for slot in range(slots):
                thread = threading.Thread(
                    target=self._worker, args=(device_id, slot),
                    name=f"mlsyslab-{device_id}-{slot}", daemon=True,
                )
                thread.start()
                threads.append(thread)

        try:
            for thread in threads:
                # Join with a timeout in a loop rather than a bare join, so Ctrl-C is
                # delivered to the main thread instead of being swallowed by the join.
                while thread.is_alive():
                    thread.join(timeout=0.5)
        except KeyboardInterrupt:
            self._stop.set()
            for thread in threads:
                thread.join(timeout=30)
            raise

        # Belt and braces. If any queue is somehow non-empty once every worker has
        # returned, those points are unmeasured, and an unmeasured point that leaves no
        # record is the one outcome this whole module exists to prevent.
        for device_id in sorted(self._queues):
            with self._cond:
                stranded = self._drain_locked(device_id)
            self._abandon(device_id, stranded,
                          "no worker remained to run it when the sweep ended")

        elapsed = time.time() - started
        summary = self._summarise(specs, elapsed, offsets)
        self._write_sidecars(summary, offsets)
        self.on_event("done", {
            "total": len(specs), "executed": len(self._records),
            "failed": sum(1 for r in self._records if not r.ok),
            "elapsed_s": elapsed, "concurrent": True,
        })
        return SweepResult(records=list(self._records), summary=summary,
                           retries=list(self._retry_log), clock_offsets=offsets)

    def _worker(self, device_id: str, slot: int) -> None:
        """One slot on one device. There are ``max_concurrency`` of these per device."""
        while True:
            job, stranded = self._next_job(device_id)
            if stranded:
                self._abandon(device_id, stranded,
                              self._breaker(device_id).reason or "device unavailable")
                continue
            if job is None:
                return

            spec, attempts = job
            try:
                self._execute(spec, attempts + 1)
            finally:
                with self._cond:
                    self._inflight -= 1
                    group = self._policy(device_id).group
                    self._group_inflight[group] = self._group_inflight.get(group, 1) - 1
                    self._cond.notify_all()

    def _next_job(self, device_id):
        """Block until this device has work, or until there is provably none left.

        Returns ``(job, stranded)``. A job means the caller now owns an in-flight slot
        and must release it. A non-empty ``stranded`` list means the device has been down
        too long and its remaining points have been taken off the queue to be recorded as
        holes. Both being empty means the sweep is over.
        """
        with self._cond:
            while True:
                if self._stop.is_set():
                    return None, []

                # Give-up check comes first. Once a device has been down through two
                # full cooldowns there is no point starting another recovery probe on
                # it, and asking that question after taking a job would mean the last
                # points of the matrix each pay a cooldown before being written off.
                if self._queues.get(device_id):
                    breaker = self._breaker(device_id)
                    if breaker.is_open and \
                            breaker.down_for(time.time()) > breaker.policy.recovery_after_s * 2:
                        stranded = self._drain_locked(device_id)
                        self._cond.notify_all()
                        return None, stranded

                job = self._take_locked(device_id)
                if job is not None:
                    self._inflight += 1
                    group = self._policy(device_id).group
                    self._group_inflight[group] = self._group_inflight.get(group, 0) + 1
                    return job, []

                # Exit only when nothing is queued anywhere *and* nobody is running.
                # A worker whose own queue drained cannot leave while another device is
                # still going, because a failover may yet hand it work.
                if self._pending_locked() == 0 and self._inflight == 0:
                    self._cond.notify_all()
                    return None, []

                self._cond.wait(timeout=0.5)

    def _execute(self, spec: RunSpec, attempt_number: int) -> None:
        context = {"spec": spec, "attempt": attempt_number,
                   "index": self._done + 1, "total": self._total}
        self.on_event("start", context)

        attempt: Attempt = self.runner.attempt(spec)
        breaker = self._breaker(spec.device_id)
        now = time.time()

        if attempt.failure_kind == "transport":
            with self._cond:
                opened = breaker.record_failure(attempt.record.error or "transport failure", now)
            self._log_retry(spec, attempt_number, attempt.record.error, opened)
            if opened:
                self.on_event("device_down", {
                    "device": spec.device_id, "reason": attempt.record.error,
                    "consecutive_failures": breaker.consecutive_failures,
                })
            if attempt_number < self.retry.max_attempts and not self._stop.is_set():
                delay = self.retry.delay_for(attempt_number + 1, self._rng)
                self.on_event("retry", {"spec": spec, "attempt": attempt_number,
                                        "delay_s": delay, "error": attempt.record.error})
                time.sleep(delay)
                self._requeue(spec.device_id, spec, attempt_number)
                return
            # Out of attempts on this device. The point is recorded as failed either
            # way; a failover queues an *additional* run elsewhere, and only where the
            # config declared the two machines interchangeable.
            self._queue_failover(spec, attempt.record.error or "transport failure")
        else:
            with self._cond:
                closed = breaker.record_success()
            if closed:
                self.on_event("device_up", {"device": spec.device_id,
                                            "recoveries": breaker.recoveries})

        self._finish(spec, attempt, attempt_number)

    def _finish(self, spec: RunSpec, attempt: Attempt, attempt_number: int) -> None:
        record = attempt.record
        if attempt_number > 1:
            # Never let a retried number look like a first-try number.
            record.warnings = list(record.warnings) + [
                f"produced on attempt {attempt_number} of {self.retry.max_attempts} "
                f"after {attempt_number - 1} transport failure(s) on {spec.device_id}"
            ]
        self.runner._save(spec, record)
        with self._cond:
            self._records.append(record)
            self._done += 1
            index = self._done
        self.on_event("finish", {"spec": spec, "record": record, "attempt": attempt_number,
                                 "index": index, "total": self._total})

    def _queue_failover(self, spec: RunSpec, reason: str) -> None:
        """Queue this point on another device, if and only if the config allows it.

        The moved point gets its own run id, because it is a different measurement, and a
        tag naming where it came from, because an analysis that cannot see the move
        cannot account for it. The original device's point is still recorded as failed,
        so the hole in the matrix stays visible in the results directory.
        """
        from .schema import compute_run_id

        policy = self._policy(spec.device_id)
        for candidate in policy.failover_to:
            if candidate == spec.device_id or self._breaker(candidate).is_open:
                continue
            moved = _respec_for_device(spec, candidate, self.runner.config)
            if moved is None:
                continue
            moved.tags = list(moved.tags) + [f"failover-from:{spec.device_id}"]
            with self._cond:
                self._queues.setdefault(candidate, []).append((moved, 0))
                self._total += 1
                self._retry_log.append({
                    "event": "failover",
                    "run_id": compute_run_id(spec.identity()),
                    "moved_run_id": compute_run_id(moved.identity()),
                    "from_device": spec.device_id, "to_device": candidate,
                    "model": spec.model, "reason": reason,
                    "note": "a moved point is a different measurement, not a repaired one",
                })
                self._cond.notify_all()
            self.on_event("failover", {"spec": spec, "to": candidate, "reason": reason})
            return

    def _drain_locked(self, device_id: str) -> List[Tuple[RunSpec, int]]:
        queue = self._queues.get(device_id) or []
        self._queues[device_id] = []
        return queue

    def _abandon(self, device_id: str, queue: List[Tuple[RunSpec, int]],
                 reason: str) -> None:
        """Record everything still queued for a dead device as failed, with the reason."""
        if not queue:
            return
        self.on_event("abandoned", {"device": device_id, "count": len(queue),
                                    "reason": reason})
        for spec, attempts in queue:
            record = self.runner._failure(
                spec,
                f"not attempted: device '{device_id}' was taken out of service after "
                f"repeated transport failures ({reason}). Re-run to retry this point."
            )
            record.warnings = list(record.warnings) + [
                "this point was never measured; the record exists so the hole in the "
                "matrix is visible and so a resumed sweep will pick it up"
            ]
            self.runner._save(spec, record)
            with self._cond:
                self._records.append(record)
                self._done += 1
                self._cond.notify_all()

    def _log_retry(self, spec: RunSpec, attempt_number: int,
                   error: Optional[str], opened: bool) -> None:
        with self._cond:
            self._retry_log.append({
                "event": "transport_failure",
                "device": spec.device_id, "model": spec.model, "mode": spec.mode,
                "attempt": attempt_number, "max_attempts": self.retry.max_attempts,
                "error": (error or "")[:600],
                "breaker_opened": bool(opened),
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })

    # --------------------------------------------------------------------- output

    def _summarise(self, specs: List[RunSpec], elapsed: float,
                   offsets: Dict[str, Any]) -> Dict[str, Any]:
        by_device: Dict[str, Dict[str, int]] = {}
        for record in self._records:
            entry = by_device.setdefault(record.device.device_id, {"ok": 0, "failed": 0})
            entry["ok" if record.ok else "failed"] += 1

        return {
            "mode": "concurrent",
            "requested": len(specs),
            "skipped_already_done": self._skipped,
            "executed": len(self._records),
            "ok": sum(1 for r in self._records if r.ok),
            "failed": sum(1 for r in self._records if not r.ok),
            "elapsed_s": round(elapsed, 2),
            "by_device": by_device,
            "concurrency": {d: p.max_concurrency for d, p in sorted(self.policies.items())},
            "resource_groups": {
                group: {"limit": self._group_limit(group),
                        "devices": sorted(d for d, p in self.policies.items()
                                          if p.group == group)}
                for group in sorted({p.group for p in self.policies.values()})
                if not group.startswith("__device__")
            },
            "retry_policy": {
                "max_attempts": self.retry.max_attempts,
                "backoff_s": self.retry.backoff_s,
                "backoff_factor": self.retry.backoff_factor,
                "max_backoff_s": self.retry.max_backoff_s,
                "retried_only": "transport failures",
            },
            "transport_failures": sum(1 for e in self._retry_log
                                      if e.get("event") == "transport_failure"),
            "breakers": {
                device_id: {"trips": b.trips, "recoveries": b.recoveries,
                            "open_at_end": b.is_open, "reason": b.reason}
                for device_id, b in sorted(self._breakers.items())
                if b.trips or b.recoveries
            },
            "clock_offsets": offsets,
        }

    def _write_sidecars(self, summary: Dict[str, Any], offsets: Dict[str, Any]) -> None:
        out = self.runner.output_dir
        with open(os.path.join(out, "schedule.json"), "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, default=str)
        if self._retry_log:
            with open(os.path.join(out, "retries.jsonl"), "a", encoding="utf-8") as fh:
                for entry in self._retry_log:
                    fh.write(json.dumps(entry, default=str) + "\n")


# ------------------------------------------------------------------------- helpers

def _respec_for_device(spec: RunSpec, device_id: str, config) -> Optional[RunSpec]:
    """A copy of a spec pointed at a different device, or None if it cannot be moved.

    A spec carries a model path that was resolved for one machine. If the config has no
    path for the new device, the point simply cannot move, and saying so is better than
    inventing one.
    """
    import copy

    entry = (config.models or {}).get(spec.model)
    if isinstance(entry, str):
        entry = {"path": entry}
    path = None
    if entry:
        path = (entry.get("paths") or {}).get(device_id) or entry.get("path")
    if spec.model not in (None, "unknown") and not path:
        return None

    moved = copy.deepcopy(spec)
    moved.device_id = device_id
    if path:
        moved.model_path = path
    return moved


def _parse_utc(stamp: str) -> Optional[float]:
    import calendar

    try:
        return calendar.timegm(time.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ"))
    except (ValueError, TypeError):
        return None


def concurrent_reporter(verbose: bool = True) -> Callable[[str, Dict[str, Any]], None]:
    """Progress printer for a concurrent sweep.

    Different from the sequential one on purpose. ``[3/40] ... ok`` on one line assumes
    the next thing printed belongs to the same run, which stops being true the moment two
    machines are working at once, so every line here names its device and stands alone.
    """
    import sys

    from .runner import _label, _metrics_line

    lock = threading.Lock()

    def report(kind: str, payload: Dict[str, Any]) -> None:
        spec = payload.get("spec")
        with lock:
            if kind == "clock":
                if payload.get("warning"):
                    sys.stderr.write(f"  clock  {payload['device']}: {payload['warning']}\n")
                elif verbose and payload.get("status") == "ok":
                    sys.stderr.write(
                        f"  clock  {payload['device']}: {payload['offset_s']:+.3f} s "
                        f"(+/- {payload['uncertainty_s']:.3f})\n")
                elif payload.get("status") == "unreachable":
                    sys.stderr.write(f"  clock  {payload['device']}: unreachable, "
                                     f"{payload.get('error')}\n")
            elif kind == "start" and verbose:
                suffix = f" attempt {payload['attempt']}" if payload.get("attempt", 1) > 1 else ""
                sys.stderr.write(f"  start  {_label(spec)}{suffix}\n")
            elif kind == "skipped" and verbose:
                sys.stderr.write(f"  skip   {_label(spec)} already done\n")
            elif kind == "retry":
                sys.stderr.write(f"  retry  {_label(spec)} in {payload['delay_s']:.1f} s: "
                                 f"{payload.get('error')}\n")
            elif kind == "device_down":
                sys.stderr.write(f"  DOWN   {payload['device']} taken out of service after "
                                 f"{payload['consecutive_failures']} transport failures: "
                                 f"{payload.get('reason')}\n")
            elif kind == "device_up":
                sys.stderr.write(f"  UP     {payload['device']} answered its recovery probe "
                                 f"and is back in service\n")
            elif kind == "failover":
                sys.stderr.write(f"  MOVED  {_label(spec)} -> {payload['to']} "
                                 f"(declared interchangeable in the config)\n")
            elif kind == "abandoned":
                sys.stderr.write(f"  HOLE   {payload['count']} point(s) on "
                                 f"{payload['device']} were never measured: "
                                 f"{payload.get('reason')}\n")
            elif kind == "finish":
                record = payload["record"]
                status = "FAILED: " + (record.error or "") if not record.ok \
                    else _metrics_line(record)
                sys.stderr.write(f"  [{payload['index']:>3}/{payload['total']}] "
                                 f"{_label(spec)}: {status}\n")
            elif kind == "done":
                sys.stderr.write(
                    f"\n{payload['executed']} run(s) in {payload['elapsed_s'] / 60:.1f} min, "
                    f"{payload['failed']} failed\n")

    return report
