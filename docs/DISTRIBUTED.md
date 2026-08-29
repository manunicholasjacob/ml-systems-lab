# Running one sweep across several machines

The sequential runner is still the reference implementation and still the default. This
document is about the other path: `mlsys run --concurrent`, which schedules a matrix
across a set of machines at once and is expected to survive one of them dying.

It is written as a design note rather than a tutorial, because the interesting part is
not the API. It is the set of places where the obvious distributed-systems answer is the
wrong answer for a benchmark, and why.

---

## What is actually being scheduled

A sweep is a list of `RunSpec`s. Each names exactly one device, because the device is the
thing being measured. That single fact changes almost every decision below: this is not a
compute cluster where work is fungible and any worker will do.

```
config.yaml ──> RunSpecs ──┬─> queue[laptop]  ──> N workers ─┐
                           ├─> queue[pi5]     ──> 1 worker  ─┼─> Runner.attempt ──> record
                           └─> queue[k8s-cpu4]──> N workers ─┘
```

One queue per device, and `max_concurrency` workers reading it. Nothing steals work from
another queue except an explicitly declared failover.

## Five decisions, and what the naive version costs

### 1. Concurrency is per device

```yaml
devices:
  laptop: { max_concurrency: 4 }
  pi5:    { max_concurrency: 1 }     # the default
```

A thread pool with one number cannot say "the laptop can take four of these and the Pi
can take exactly one". The 2 GB Pi has hung hard enough to need a physical power cycle
when given two heavy jobs at once. The default is 1 everywhere: guessing high costs a
corrupted measurement or a wedged board, guessing low costs wall-clock time.

### 2. Some devices are the same machine

```yaml
devices:
  wsl-host:      { resource_group: laptop-cpu }
  k8s-unlimited: { resource_group: laptop-cpu }
  k8s-cpu4:      { resource_group: laptop-cpu }
```

A host and the pods scheduled onto it are three device ids and one CPU. Running them at
once is not a faster sweep, it is three measurements of contention. Devices in a
`resource_group` hold one job at a time between them, unless `group_max_concurrency` says
otherwise, and the tightest limit any member declares wins.

The group also takes turns. First come, first served turns out to mean "whichever worker
reacquires the lock after releasing it", which is reliably the one that just finished; in
a five-arm study on one CPU that let one arm run its entire queue before any other arm
started, putting each arm in a different part of the machine's thermal history. Round
robin across the group is an experimental control, not a fairness nicety.

### 3. Only transport failures are retried

`Runner.attempt` returns a record *and* a classification:

| kind | what it means | retried |
|---|---|---|
| `transport` | a `DeviceError`: ssh dropped, the pod went away, the agent never answered | yes |
| `measurement` | the run happened and came back saying no, e.g. the model did not fit | **no** |
| `config` | our own code could not build the task | no |
| `parse` | the run happened and the parser broke; raw output is kept | no |

A model that will not fit in RAM is a *result*. Retrying it three times turns a finding
into a hole in the matrix with nothing to show for it. This distinction is the reason
`attempt()` exists alongside `run_one()`.

Two places got this wrong and were fixed, both in the same direction:

* `exists()` used to return False when it could not run the check at all. A one-second
  outage during backend discovery came back as "llama-bench not found on k8s-cpu2", was
  classified `config`, and was never retried. It now raises when it cannot tell, because
  "no" and "I could not look" are different answers.
* A `DeviceError` raised while *building* a task was classified `config`. Building a task
  talks to the device, so it now classifies as `transport`.

### 4. Every retry is written down

```
runs/<experiment>/retries.jsonl     one line per transport failure and failover
runs/<experiment>/schedule.json     policy, per-device outcome, breaker trips, clock offsets
```

and the record itself carries a warning:

```json
"warnings": ["produced on attempt 2 of 3 after 1 transport failure(s) on pi5"]
```

A benchmark that silently retries is a benchmark you cannot trust, because the reader has
no way to know whether a number came from a clean run or the third attempt on a machine
that was already misbehaving.

### 5. Work is not moved between devices

In most schedulers, requeuing a task elsewhere is the obvious response to a dead worker.
Here it is close to the worst available option: a Pi measurement taken on the laptop is
not a repaired data point, it is a different experiment wearing the same run id.

So the default is to leave a hole, and to make the hole loud:

```json
{"status": "failed",
 "error": "not attempted: device 'pi5' was taken out of service after repeated transport failures (...). Re-run to retry this point.",
 "warnings": ["this point was never measured; the record exists so the hole in the matrix is visible and so a resumed sweep will pick it up"]}
```

Failover happens only where a config declares two devices interchangeable:

```yaml
devices:
  pi5: { failover_to: [pi5-spare] }
```

and even then the moved point gets its own run id and a `failover-from:pi5` tag, because
it is a different measurement and no analysis should be able to mistake it.

## Partial failure, concretely

Per device, a circuit breaker with three states:

```
closed ──(N consecutive transport failures)──> open
open   ──(recovery_after_s elapsed)──────────> half-open: exactly one probe
half-open ──(probe succeeded)────────────────> closed
          ──(probe failed)───────────────────> open, cooldown restarts
```

Without it, one unreachable machine costs the sweep its full retry budget on every
remaining point on that machine. With it, the device is taken out of service and given
one chance to come back later, which is what turns a tailnet that drops for two minutes
into a delay rather than a lost night.

`down_since` is tracked separately from `opened_at` for a reason: `opened_at` restarts on
every failed probe, so a "has this device been down too long" test built on it would
never fire. After two full cooldowns with no successful run, the device's remaining
points are drained and recorded as holes, and the sweep finishes on the machines that are
still answering.

This is not only tested against fakes. On the containerisation sweep, `k8s-cpu2` took
three transport failures in a row, the breaker opened, the recovery probe succeeded two
minutes later, and the run completed:

```
DOWN   k8s-cpu2 taken out of service after 3 transport failures: ...
UP     k8s-cpu2 answered its recovery probe and is back in service
[  5/5] k8s-cpu2 qwen0.5b-q8 throughput t4 p128/n64: prefill 56.8 t/s, decode 15.56 t/s
5 run(s) in 5.4 min, 0 failed
```

## Resume

Run ids are deterministic over the spec, so a killed sweep restarts with no duplicate
work. `already_done()` treats a record as complete only if it exists and its status is
`ok`; failures are retried, because a failure is usually environmental and re-running is
cheap next to carrying a silent hole.

The same run above is what resume looks like in practice: 20 of 25 points were already on
disk, the resumed sweep executed exactly the 5 that were missing, and the completed ones
were not touched.

## Clocks

Before a concurrent sweep, every device is asked what time it is, and the offset is
recorded in `schedule.json` whether or not it is alarming. Cross-device timing compared
after the fact is only as good as the worst clock in the set, and a board whose NTP never
came up can be minutes out with nothing looking wrong.

The estimate is Cristian's algorithm: the midpoint of the agent's own execution minus the
midpoint of the round trip. The agent reports both when it started and when it finished
precisely so that its own runtime cancels. An earlier version used only the finishing
stamp and reported four devices on one physical machine as two to three seconds apart,
which is nonsense, because they share a clock; the bias was the two seconds the agent
spends collecting sysinfo. The uncertainty reported alongside is the transport time that
is *not* accounted for, so a device whose offset falls inside its own uncertainty is
reported as agreeing rather than as slightly wrong.

## Watching it

```
mlsys run configs/lab.yaml --concurrent --prometheus-port 9109
mlsys export runs/lab --serve            # a finished directory, same metrics
```

`grafana/mlsyslab-sweep.json` is a dashboard for it: devices up or down, clock offsets,
effective CPU quotas, runs finished and in flight per device, and throughput, power and
temperature per point as each one completes.

Three rules keep this a window rather than a second, worse copy of the science:

* **The exporter is never in the write path.** Every number reaches it after being
  written to disk, so a scrape that never happens costs a graph, not a measurement.
* **No new dependency.** The text exposition format is about fifty lines of code, and
  the alternative is adding `prometheus_client` to the dependency list of a package whose
  measurement core has to run on a 2 GB board.
* **Labels are bounded, and complete.** Device, backend, model, quantisation, mode,
  threads and token shape are all bounded by the config. Run ids and timestamps are never
  labels. Leaving out a swept axis would not save much cardinality; it would silently
  collapse points that were never the same measurement onto one line of a graph, which is
  the specific way a dashboard starts lying.

What it is not: a live power trace. Telemetry is sampled on the device, inside the agent,
and comes back when the run does. A point appears when it finishes.

## What this does not demonstrate

Three machines in a home lab is not fleet scale, and nothing here should be read as
evidence about fleet-scale behaviour. What it does show is an implementation against a
container orchestrator, partial-failure handling that has been exercised by real machines
failing rather than only by mocks, and a measurement that stays valid across devices.
Those are the claims; they are worth more than a larger one that does not hold.
