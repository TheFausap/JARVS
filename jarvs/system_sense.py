"""
system_sense.py — JARVS's awareness of the machine it runs on (Ubuntu/Linux).

Reads live host telemetry straight from the OS: CPU load/utilization, memory, NVIDIA
GPU load/VRAM/temperature (via ``nvidia-smi``), disk, thermals, uptime, the local
clock/timezone, and approximate geolocation from the router's public IP. Everything is
best-effort and degradable — a missing source returns ``None`` rather than raising, so
this never breaks a reply.

Geolocation is the one piece that reaches the network: it calls an IP-geo service
(cached for ``config.GEO_TTL``) and is gated by ``config.ALLOW_WEB`` and
``config.GEOLOCATION``. It reveals the public IP to that service, so it is opt-out.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import subprocess
import time
from datetime import datetime, timezone

import requests

from . import config


# --- CPU ---------------------------------------------------------------------- #

def _read_stat() -> tuple[int, int] | None:
    try:
        with open("/proc/stat", "r") as fh:
            parts = fh.readline().split()
        vals = [int(x) for x in parts[1:]]
        idle = vals[3] + (vals[4] if len(vals) > 4 else 0)   # idle + iowait
        return sum(vals), idle
    except (OSError, ValueError, IndexError):
        return None


def cpu(sample: float = 0.12) -> dict | None:
    """CPU load averages, core count, and an instantaneous utilization % sampled
    from two /proc/stat reads `sample` seconds apart."""
    try:
        with open("/proc/loadavg", "r") as fh:
            la = fh.read().split()[:3]
        load1, load5, load15 = (float(la[0]), float(la[1]), float(la[2]))
    except (OSError, ValueError, IndexError):
        load1 = load5 = load15 = None
    cores = os.cpu_count() or 1
    util = None
    a = _read_stat()
    if a:
        time.sleep(sample)
        b = _read_stat()
        if b and b[0] != a[0]:
            total_d, idle_d = b[0] - a[0], b[1] - a[1]
            util = round(100.0 * (total_d - idle_d) / total_d, 1) if total_d else None
    return {"load1": load1, "load5": load5, "load15": load15,
            "cores": cores, "util_pct": util}


# --- memory ------------------------------------------------------------------- #

def memory() -> dict | None:
    info = {}
    try:
        with open("/proc/meminfo", "r") as fh:
            for line in fh:
                k, _, rest = line.partition(":")
                info[k] = int(rest.strip().split()[0])  # kB
    except (OSError, ValueError, IndexError):
        return None
    total = info.get("MemTotal", 0)
    avail = info.get("MemAvailable", info.get("MemFree", 0))
    used = total - avail
    return {"total_mb": total // 1024, "available_mb": avail // 1024,
            "used_mb": used // 1024,
            "used_pct": round(100.0 * used / total, 1) if total else None}


# --- GPU (NVIDIA) ------------------------------------------------------------- #

def gpu() -> list[dict]:
    """Per-GPU utilization/VRAM/temperature via nvidia-smi. Empty list when there is
    no NVIDIA GPU or the tool is absent."""
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        if out.returncode != 0:
            return []
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in out.stdout.strip().splitlines():
        cols = [c.strip() for c in line.split(",")]
        if len(cols) < 5:
            continue
        def _num(x):
            try:
                return int(float(x))
            except ValueError:
                return None
        gpus.append({"name": cols[0], "util_pct": _num(cols[1]),
                     "mem_used_mb": _num(cols[2]), "mem_total_mb": _num(cols[3]),
                     "temp_c": _num(cols[4])})
    return gpus


# --- disk / thermals / time / host / net -------------------------------------- #

def disk(path: str = "/") -> dict | None:
    try:
        u = shutil.disk_usage(path)
    except OSError:
        return None
    return {"total_gb": round(u.total / 1e9, 1), "used_gb": round(u.used / 1e9, 1),
            "free_gb": round(u.free / 1e9, 1),
            "used_pct": round(100.0 * u.used / u.total, 1) if u.total else None}


def cpu_temp() -> float | None:
    """Best CPU/package temperature from /sys/class/thermal (°C), if exposed."""
    best = None
    try:
        base = "/sys/class/thermal"
        for zone in os.listdir(base):
            if not zone.startswith("thermal_zone"):
                continue
            try:
                with open(os.path.join(base, zone, "type")) as fh:
                    ztype = fh.read().strip().lower()
                with open(os.path.join(base, zone, "temp")) as fh:
                    milli = int(fh.read().strip())
            except (OSError, ValueError):
                continue
            c = milli / 1000.0
            # prefer a package/core sensor; otherwise keep the max seen
            if "x86_pkg" in ztype or "coretemp" in ztype or "k10temp" in ztype:
                return round(c, 1)
            best = c if best is None else max(best, c)
    except OSError:
        return None
    return round(best, 1) if best is not None else None


def _human_duration(secs: float) -> str:
    secs = int(secs)
    d, rem = divmod(secs, 86400)
    h, rem = divmod(rem, 3600)
    m, _ = divmod(rem, 60)
    parts = [f"{d}d" if d else "", f"{h}h" if h else "", f"{m}m"]
    return " ".join(p for p in parts if p)


def time_now() -> dict:
    local = datetime.now().astimezone()
    uptime = None
    try:
        with open("/proc/uptime") as fh:
            uptime = float(fh.read().split()[0])
    except (OSError, ValueError, IndexError):
        pass
    return {
        "iso": local.isoformat(timespec="seconds"),
        "local": local.strftime("%A %Y-%m-%d %H:%M:%S"),
        "tz": str(local.tzinfo),
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "uptime": _human_duration(uptime) if uptime is not None else None,
    }


def host() -> dict:
    pretty = ""
    try:
        with open("/etc/os-release") as fh:
            for line in fh:
                if line.startswith("PRETTY_NAME="):
                    pretty = line.split("=", 1)[1].strip().strip('"')
                    break
    except OSError:
        pass
    return {"hostname": socket.gethostname(), "os": pretty or platform.platform(),
            "kernel": platform.release(), "arch": platform.machine(),
            "python": platform.python_version()}


def network() -> dict | None:
    """Local IP, default gateway, and interface, parsed from `ip route get`."""
    try:
        out = subprocess.run(["ip", "route", "get", "1.1.1.1"],
                             capture_output=True, text=True, timeout=5)
        toks = out.stdout.split()
    except (OSError, subprocess.SubprocessError):
        return None
    info = {}
    for key, field in (("via", "gateway"), ("dev", "interface"), ("src", "local_ip")):
        if key in toks:
            try:
                info[field] = toks[toks.index(key) + 1]
            except IndexError:
                pass
    return info or None


# --- geolocation (network, cached) -------------------------------------------- #

def _geo_cache_path() -> str:
    return os.path.join(str(config.DATA_DIR), "geo_cache.json")


def _load_geo_cache() -> dict | None:
    try:
        with open(_geo_cache_path()) as fh:
            data = json.load(fh)
        if time.time() - float(data.get("ts", 0)) < config.GEO_TTL:
            data["cached"] = True
            return data
    except (OSError, json.JSONDecodeError, ValueError):
        pass
    return None


def geolocation(force: bool = False) -> dict | None:
    """Approximate location from the router's public IP (city-level), cached for
    config.GEO_TTL. None when disabled, offline, or the lookup fails."""
    if not (config.GEOLOCATION and config.ALLOW_WEB):
        return None
    if not force:
        cached = _load_geo_cache()
        if cached:
            return cached
    try:
        r = requests.get(config.GEO_URL, timeout=6)
        r.raise_for_status()
        d = r.json()
    except Exception:  # noqa: BLE001
        return _load_geo_cache()   # fall back to a stale cache if we have one
    if isinstance(d, dict) and d.get("status") == "success":
        geo = {"city": d.get("city"), "region": d.get("regionName"),
               "country": d.get("country"), "lat": d.get("lat"), "lon": d.get("lon"),
               "timezone": d.get("timezone"), "isp": d.get("isp"),
               "public_ip": d.get("query"), "ts": time.time(), "cached": False}
        try:
            os.makedirs(str(config.DATA_DIR), exist_ok=True)
            with open(_geo_cache_path(), "w") as fh:
                json.dump(geo, fh)
        except OSError:
            pass
        return geo
    return _load_geo_cache()


# --- aggregate + rendering ---------------------------------------------------- #

def snapshot(include_geo: bool = False) -> dict:
    snap = {"time": time_now(), "cpu": cpu(), "memory": memory(), "gpu": gpu(),
            "disk": disk(), "cpu_temp_c": cpu_temp(), "host": host(),
            "network": network()}
    if include_geo:
        snap["geo"] = geolocation()
    return snap


def summary_block(include_geo: bool = False) -> str:
    """A compact ``[SYSTEM STATUS]`` block for grounding a reply in real telemetry."""
    s = snapshot(include_geo=include_geo)
    lines = ["[SYSTEM STATUS] (live readings from this machine — use for any question "
             "about the host, time, or location; do not invent numbers)"]
    t = s.get("time") or {}
    if t.get("local"):
        lines.append(f"- Time: {t['local']} ({t.get('tz','')}); uptime {t.get('uptime','?')}")
    c = s.get("cpu") or {}
    if c:
        lines.append(f"- CPU: {c.get('util_pct','?')}% across {c.get('cores','?')} cores "
                     f"(load {c.get('load1','?')}/{c.get('load5','?')}/{c.get('load15','?')})"
                     + (f", {s['cpu_temp_c']}°C" if s.get("cpu_temp_c") is not None else ""))
    m = s.get("memory") or {}
    if m:
        lines.append(f"- Memory: {m.get('used_mb','?')} / {m.get('total_mb','?')} MB "
                     f"used ({m.get('used_pct','?')}%)")
    for i, g in enumerate(s.get("gpu") or []):
        lines.append(f"- GPU{i}: {g.get('name','?')} — {g.get('util_pct','?')}% util, "
                     f"{g.get('mem_used_mb','?')}/{g.get('mem_total_mb','?')} MB, "
                     f"{g.get('temp_c','?')}°C")
    d = s.get("disk") or {}
    if d:
        lines.append(f"- Disk /: {d.get('free_gb','?')} GB free of {d.get('total_gb','?')} GB")
    h = s.get("host") or {}
    if h:
        lines.append(f"- Host: {h.get('hostname','?')} · {h.get('os','?')} · "
                     f"kernel {h.get('kernel','?')}")
    n = s.get("network") or {}
    if n:
        lines.append(f"- Network: {n.get('local_ip','?')} via {n.get('gateway','?')} "
                     f"({n.get('interface','?')})")
    g = s.get("geo")
    if g:
        where = ", ".join(x for x in (g.get("city"), g.get("region"), g.get("country")) if x)
        stale = " (cached)" if g.get("cached") else ""
        lines.append(f"- Location: {where}{stale} — ~{g.get('lat')},{g.get('lon')} "
                     f"via {g.get('isp','?')} (IP-based, approximate)")
    return "\n".join(lines)


import re

_SYS_RE = re.compile(
    r"\b(?:cpu|gpu|vram|ram|memory usage|mem(?:ory)? (?:load|free|used)|load average|"
    r"system (?:load|status|resources?)|hardware|disk space|disk usage|uptime|"
    r"temperature|how hot|fan|this (?:machine|computer|box|host|server|pc)|"
    r"the (?:machine|host|server)|your (?:hardware|machine|host)|specs?|"
    r"what time|current time|the time|what day|what(?:'s| is) the date|today's date)\b",
    re.I)
_LOC_RE = re.compile(
    r"\b(?:where (?:am i|are we|is this|are you)|my location|our location|geolocation|"
    r"what city|which city|what country|which country|my city|located)\b", re.I)


def needs_system_info(text: str) -> bool:
    return bool(_SYS_RE.search(text or "")) or needs_location(text)


def needs_location(text: str) -> bool:
    return bool(_LOC_RE.search(text or ""))


def status() -> str:
    g = gpu()
    gpu_str = (f"{len(g)} GPU" if g else "no GPU")
    geo = "geo on" if (config.GEOLOCATION and config.ALLOW_WEB) else "geo off"
    return f"system-sense: on ({os.cpu_count()} cores, {gpu_str}, {geo})"
