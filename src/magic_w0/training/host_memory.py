"""Per-process host-memory probe, for attributing a node-level memory climb.

WHY THIS EXISTS.  The 48-GPU pretrain was OOMKilled after ~60 hours (EIP 7370,
step 44760) with the node's memory climbing ~2.5 GB/h, and diagnosing it took
days largely because there was nothing to look at. W&B's system metrics give
exactly two host numbers -- the node's available memory and rank 0's main-process
RSS -- for a node running 8 ranks and 256 dataloader workers. Two observables for
256 processes cannot attribute anything, and the one process that *was* visible
(rank 0's main) stayed flat the whole time, which is what made the climb look
like it came from nowhere. It came from the workers.

Nor does W&B see the other five nodes: it monitors whichever node rank 0 landed
on, so the node that actually runs out first can be invisible right up to the
kill.

WHAT IT ANSWERS.  One question, which is the one that unblocks everything else:
*which class of process is growing* -- the main process, the dataloader workers
as a group, or a single runaway worker. Everything downstream (which loader,
which source, which read) follows from that answer, and none of it follows
without it.

COST AND SAFETY.  Reading /proc for ~32 children is well under a millisecond and
happens only on log steps, but this is off unless ``MAGIC_W0_HOST_MEM_PROBE=1``
because it is diagnostic weight on a production loop, not something to carry
forever. Every read is guarded: /proc can be restricted or a child can exit
mid-walk, and a memory probe must never be the thing that kills a multi-day run.
"""

from __future__ import annotations

import os

ENV_FLAG = "MAGIC_W0_HOST_MEM_PROBE"
CGROUP_V2_ROOT = "/sys/fs/cgroup"
CGROUP_V1_ROOT = "/sys/fs/cgroup/memory"


def probe_enabled() -> bool:
    """Opt-in, because this is diagnostic weight on a production training loop."""
    return os.environ.get(ENV_FLAG, "").strip().lower() in {"1", "true", "yes"}


def _rss_bytes(pid: str | int = "self") -> int:
    """Physical memory attributable to one process, or 0 if it is unreadable.

    Pss, not VmRSS, and the difference is the whole point once these are summed
    over 32 dataloader workers. VmRSS counts every resident page the process maps,
    shared or not, so a parquet page sitting in the page cache and read by all 32
    workers is counted 32 times. Summing VmRSS therefore reported 156 GB of worker
    memory per rank, which cannot be true: eight ranks would be 1,250 GB on a
    1,008 GB node. Pss divides each shared page by the number of processes mapping
    it, so a sum of Pss over a process tree is the physical memory that tree
    actually costs -- which is the number the OOM killer is working from.

    smaps_rollup is a kernel VMA walk rather than a field lookup, so it costs more
    than reading status; that is why the probe as a whole runs only on log steps.
    Falls back to VmRSS where smaps_rollup is absent (pre-4.14 kernels, or a
    hardened /proc), because a coarse number beats no number -- but the fallback
    over-counts sharing, so a run that used it is not comparable with one that did
    not.
    """
    try:
        with open(f"/proc/{pid}/smaps_rollup") as handle:
            for line in handle:
                if line.startswith("Pss:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    try:
        with open(f"/proc/{pid}/status") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


def _child_pids() -> list[str]:
    """PIDs whose parent is this process: the dataloader workers.

    Walks /proc rather than /proc/self/task/*/children because the latter is not
    present on every kernel config. All ranks on a node share a PID namespace, so
    the ppid filter is what keeps a rank from counting its neighbours' workers.
    """
    me = os.getpid()
    children = []
    try:
        entries = os.listdir("/proc")
    except OSError:
        return children
    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat") as handle:
                fields = handle.read()
        except OSError:  # the process exited between listdir and open
            continue
        # comm (field 2) is parenthesised and may contain spaces, so split after
        # its closing paren: ppid is then the second field of the remainder.
        close = fields.rfind(")")
        if close < 0:
            continue
        rest = fields[close + 2 :].split()
        if len(rest) >= 2 and rest[1] == str(me):
            children.append(entry)
    return children


def _available_bytes() -> int:
    """Node-level MemAvailable, or 0 where /proc/meminfo is not readable.

    Logged per rank so that reducing MIN across ranks surfaces the node closest to
    the OOM killer -- not merely the node W&B happens to watch.

    ONLY MEANINGFUL ON A WHOLE NODE. This reads the machine, not the job: on a
    partial-node slice (an ml.a800.4 on an 8-GPU box) a neighbouring tenant moves
    it as much as this job does. Measured on EIP 7822, it fell 105 GB in 74
    seconds while this job's own memory moved 0.1 GB, and bounced between 743 and
    906 GB thereafter. Use _cgroup_used_bytes() there instead.
    """
    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


def _cgroup_used_bytes() -> int:
    """This container's own memory charge, or 0 if no cgroup file is readable.

    Unlike MemAvailable this is unaffected by anything else on the machine, and it
    is what the OOM killer acts on: a task is killed when its cgroup hits its
    limit, not when the node runs low. Tries cgroup v2 then v1.
    """
    for path in (
        os.path.join(CGROUP_V2_ROOT, "memory.current"),
        os.path.join(CGROUP_V1_ROOT, "memory.usage_in_bytes"),
    ):
        try:
            with open(path) as handle:
                return int(handle.read().strip())
        except (OSError, ValueError):
            continue
    return 0


def _read_cgroup_int(path: str) -> int | None:
    """Read one cgroup integer; ``max`` and unreadable files mean unavailable."""
    try:
        with open(path) as handle:
            value = handle.read().strip()
        if value == "max":
            return None
        return int(value)
    except (OSError, ValueError):
        return None


def _cgroup_limits_bytes() -> dict[str, int | None]:
    """Return finite hard/high memory limits where the runtime exposes them."""
    current = os.path.join(CGROUP_V2_ROOT, "memory.current")
    if os.path.exists(current):
        return {
            "max": _read_cgroup_int(os.path.join(CGROUP_V2_ROOT, "memory.max")),
            "high": _read_cgroup_int(os.path.join(CGROUP_V2_ROOT, "memory.high")),
        }

    maximum = _read_cgroup_int(os.path.join(CGROUP_V1_ROOT, "memory.limit_in_bytes"))
    high = _read_cgroup_int(os.path.join(CGROUP_V1_ROOT, "memory.soft_limit_in_bytes"))
    # v1 represents an unlimited value as a huge integer close to LONG_MAX.
    return {
        "max": maximum if maximum is not None and maximum < (1 << 60) else None,
        "high": high if high is not None and high < (1 << 60) else None,
    }


def _cgroup_key_values(filename: str) -> dict[str, int]:
    """Read a cgroup v2 ``key value`` file without ever raising."""
    result: dict[str, int] = {}
    try:
        with open(os.path.join(CGROUP_V2_ROOT, filename)) as handle:
            for line in handle:
                parts = line.split()
                if len(parts) != 2:
                    continue
                try:
                    result[parts[0]] = int(parts[1])
                except ValueError:
                    continue
    except OSError:
        return {}
    return result


def _cgroup_stat() -> dict[str, int]:
    """Read selected cgroup v2 memory.stat gauges and counters.

    The memory gauges are bytes. Reclaim/fault fields are cumulative counters;
    their slope in W&B is useful even when a terminal stall prevents another
    training step from completing.
    """
    wanted = {
        "anon",
        "file",
        "inactive_file",
        "active_file",
        "shmem",
        "pgmajfault",
        "workingset_refault_file",
        "pgscan",
        "pgsteal",
    }
    return {
        key: value
        for key, value in _cgroup_key_values("memory.stat").items()
        if key in wanted
    }


def _cgroup_events() -> dict[str, int]:
    """Cumulative memory.high/max/OOM event counters for cgroup v2."""
    wanted = {"high", "max", "oom", "oom_kill"}
    return {
        key: value
        for key, value in _cgroup_key_values("memory.events").items()
        if key in wanted
    }


def _cgroup_pressure() -> dict[str, float]:
    """Return cgroup memory PSI averages, or an empty dict when unavailable."""
    result: dict[str, float] = {}
    try:
        with open(os.path.join(CGROUP_V2_ROOT, "memory.pressure")) as handle:
            for line in handle:
                parts = line.split()
                if not parts or parts[0] not in {"some", "full"}:
                    continue
                for item in parts[1:]:
                    name, separator, raw = item.partition("=")
                    if separator and name in {"avg10", "avg60"}:
                        try:
                            result[f"{parts[0]}_{name}"] = float(raw)
                        except ValueError:
                            continue
    except OSError:
        return {}
    return result


def _dev_shm_bytes() -> tuple[int, int] | None:
    """Return used/free bytes in /dev/shm, where DataLoader IPC is allocated."""
    try:
        stat = os.statvfs("/dev/shm")
    except OSError:
        return None
    used = (stat.f_blocks - stat.f_bfree) * stat.f_frsize
    free = stat.f_bavail * stat.f_frsize
    return used, free


def sample() -> dict[str, float | None]:
    """This rank's host-memory picture, gauges in GiB and counters as totals.

    Returns zeros for required gauges and None for unavailable optional metrics.
    A caller that reduces these across ranks must still use a fixed key list so a
    node with fewer cgroup files cannot make ranks enter different collectives.
    """
    children = _child_pids()
    worker_rss = [_rss_bytes(pid) for pid in children]
    cgroup_used = _cgroup_used_bytes()
    cgroup_limits = _cgroup_limits_bytes()
    cgroup_stat = _cgroup_stat()
    cgroup_events = _cgroup_events()
    cgroup_pressure = _cgroup_pressure()
    dev_shm = _dev_shm_bytes()
    gib = 1024**3
    result: dict[str, float | None] = {
        "host_rss_main_gb": _rss_bytes() / gib,
        "host_rss_workers_gb": sum(worker_rss) / gib,
        "host_rss_worker_max_gb": (max(worker_rss) if worker_rss else 0) / gib,
        "host_worker_count": float(len(worker_rss)),
        "host_avail_gb": _available_bytes() / gib,
        "host_cgroup_used_gb": cgroup_used / gib,
    }

    maximum = cgroup_limits.get("max")
    high = cgroup_limits.get("high")
    result["host_cgroup_max_gb"] = maximum / gib if maximum is not None else None
    result["host_cgroup_high_gb"] = high / gib if high is not None else None
    result["host_cgroup_headroom_gb"] = (
        max(maximum - cgroup_used, 0) / gib
        if maximum is not None and cgroup_used > 0
        else None
    )
    result["host_cgroup_high_headroom_gb"] = (
        max(high - cgroup_used, 0) / gib
        if high is not None and cgroup_used > 0
        else None
    )

    for field in ("anon", "file", "inactive_file", "active_file", "shmem"):
        value = cgroup_stat.get(field)
        result[f"host_cgroup_{field}_gb"] = value / gib if value is not None else None
    for field in ("pgmajfault", "workingset_refault_file", "pgscan", "pgsteal"):
        value = cgroup_stat.get(field)
        result[f"host_cgroup_{field}_total"] = (
            float(value) if value is not None else None
        )
    for field in ("high", "max", "oom", "oom_kill"):
        value = cgroup_events.get(field)
        result[f"host_cgroup_events_{field}_total"] = (
            float(value) if value is not None else None
        )
    for field in ("some_avg10", "some_avg60", "full_avg10", "full_avg60"):
        result[f"host_cgroup_psi_{field}"] = cgroup_pressure.get(field)

    result["host_dev_shm_used_gb"] = dev_shm[0] / gib if dev_shm is not None else None
    result["host_dev_shm_free_gb"] = dev_shm[1] / gib if dev_shm is not None else None
    return result


__all__ = ["ENV_FLAG", "probe_enabled", "sample"]
