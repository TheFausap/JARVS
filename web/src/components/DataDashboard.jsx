import { motion } from "framer-motion";
import {
  Activity, Cpu, MemoryStick, Thermometer, Gauge, ListChecks,
  Volume2, VolumeX, Radio, MapPin, Clock, Repeat, Play, Pause, Trash2,
  Code2, Coins,
} from "lucide-react";

// Dot colour per Claude-exchange outcome.
const CLAUDE_DOT = {
  done: "#3ddc97",
  needs_clarification: "#f5a623",
  held: "#f5a623",
  reject: "#e5484d",
  rejected_local: "#e5484d",
  over_budget: "#e5484d",
  error: "#e5484d",
};

function nextIn(nextRun) {
  const s = Math.max(0, Math.round(nextRun - Date.now() / 1000));
  if (s >= 3600) return `${Math.floor(s / 3600)}h${Math.floor((s % 3600) / 60)}m`;
  if (s >= 60) return `${Math.floor(s / 60)}m`;
  return `${s}s`;
}

function schedLabel(r) {
  const s = r.interval_sec;
  let base = `every ${s}s`;
  if (s % 604800 === 0) base = s === 604800 ? "weekly" : `every ${s / 604800} weeks`;
  else if (s % 86400 === 0) base = s === 86400 ? "daily" : `every ${s / 86400} days`;
  else if (s % 3600 === 0) base = s === 3600 ? "hourly" : `every ${s / 3600} hours`;
  else if (s % 60 === 0) base = `every ${s / 60} min`;
  if (r.at_hour !== null && r.at_hour !== undefined) {
    base += ` at ${String(r.at_hour).padStart(2, "0")}:${String(r.at_minute || 0).padStart(2, "0")}`;
  }
  return base;
}

function IconBtn({ icon: Icon, title, onClick, color = "#8a94a3" }) {
  return (
    <button title={title} onClick={onClick}
      className="rounded p-1 transition-colors hover:bg-white/5">
      <Icon size={12} color={color} strokeWidth={1.6} />
    </button>
  );
}

const JOB_DOT = {
  queued: "#8a94a3", running: "#00d4ff", completed: "#3ddc97", failed: "#e5484d",
};

function Widget({ icon: Icon, label, value, unit, accent }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
      className="hairline rounded-md bg-[#0e0f12]/70 px-3 py-2.5"
    >
      <div className="flex items-center gap-2">
        <Icon size={13} color={accent ? "#00d4ff" : "#8a94a3"} strokeWidth={1.5} />
        <span className="font-mono text-[9px] tracking-[0.22em] text-slate-500">{label}</span>
      </div>
      <div className="mt-1.5 flex items-baseline gap-1">
        <span className="font-mono text-lg" style={{ color: accent ? "#00d4ff" : "#d7dbe3" }}>
          {value}
        </span>
        {unit && <span className="font-mono text-[10px] text-slate-600">{unit}</span>}
      </div>
    </motion.div>
  );
}

function SectionLabel({ children }) {
  return (
    <span className="font-mono text-[9px] tracking-[0.3em] text-slate-600">{children}</span>
  );
}

const dash = (v) => (v === null || v === undefined ? "—" : v);

export default function DataDashboard({ stats, status, jobs = [], system, routines = [], claude = null, onRoutineAction, onToggleVoice }) {
  const online = status?.backend_ok;
  const voiceOn = stats?.voice_enabled ?? status?.voice_enabled;
  const jobCounts = stats?.jobs || {};
  const activeJobs = (jobCounts.queued || 0) + (jobCounts.running || 0);

  const cpu = system?.cpu || {};
  const mem = system?.memory || {};
  const gpu = (system?.gpu && system.gpu[0]) || {};
  const budget = claude?.budget;
  const exchanges = claude?.exchanges || [];
  const frac = budget?.cap ? Math.min(1, budget.frac || 0) : 0;
  const barColor = budget?.over ? "#e5484d" : frac >= 0.8 ? "#f5a623" : "#00d4ff";
  const fmt = (n) => (n ?? 0).toLocaleString();

  const geo = system?.geo;
  const clock = system?.time?.local;
  const where = geo
    ? [geo.city, geo.country].filter(Boolean).join(", ")
    : null;

  return (
    <aside className="hairline flex w-72 shrink-0 flex-col gap-4 border-y-0 border-r-0 bg-[#0b0c0e]/60 p-5 backdrop-blur-sm">
      {/* heartbeat */}
      <div className="flex items-center justify-between">
        <span className="font-mono text-[10px] tracking-[0.3em] text-slate-400">SYSTEM</span>
        <span className="flex items-center gap-1.5 font-mono text-[10px]">
          <Radio size={11} color={online ? "#00d4ff" : "#e5484d"} strokeWidth={2} />
          <span style={{ color: online ? "#00d4ff" : "#e5484d" }}>
            {online ? "ONLINE" : "OFFLINE"}
          </span>
        </span>
      </div>

      {/* live clock */}
      {clock && (
        <div className="flex items-center gap-2 font-mono text-[11px] text-slate-400">
          <Clock size={12} color="#8a94a3" strokeWidth={1.5} />
          <span>{clock}</span>
        </div>
      )}

      {/* host telemetry */}
      <div className="flex flex-col gap-2">
        <SectionLabel>HOST</SectionLabel>
        <div className="grid grid-cols-2 gap-2.5">
          <Widget icon={Cpu} label="CPU" value={dash(cpu.util_pct)} unit="%" accent />
          <Widget icon={Gauge} label="GPU" value={dash(gpu.util_pct)} unit="%" accent />
          <Widget icon={MemoryStick} label="RAM" value={dash(mem.used_pct)} unit="%" />
          <Widget icon={Thermometer} label="GPU °C" value={dash(gpu.temp_c)} />
        </div>
      </div>

      {/* JARVS internals */}
      <div className="flex flex-col gap-2">
        <SectionLabel>JARVS</SectionLabel>
        <div className="grid grid-cols-2 gap-2.5">
          <Widget icon={Activity} label="LATENCY" value={dash(stats?.latency_ms)} unit="ms" accent />
          <Widget icon={ListChecks} label="JOBS" value={activeJobs} accent={activeJobs > 0} />
          <Widget icon={Activity} label="TURNS" value={dash(stats?.turns)} />
          <Widget icon={Activity} label="TOKENS" value={dash(stats?.last_tokens)} />
        </div>
      </div>

      {/* recurring routines */}
      {routines.length > 0 && (
        <div className="hairline rounded-md bg-[#0e0f12]/70 px-3 py-2.5">
          <div className="flex items-center gap-2">
            <Repeat size={12} color="#00d4ff" strokeWidth={1.5} />
            <SectionLabel>ROUTINES</SectionLabel>
          </div>
          <div className="mt-2 flex flex-col gap-2.5">
            {routines.map((r) => (
              <div key={r.id} className="flex items-start gap-2">
                <span className="mt-1 h-1.5 w-1.5 shrink-0 rounded-full"
                      style={{ background: r.enabled ? "#00d4ff" : "#4b5563" }} />
                <div className="flex min-w-0 flex-1 flex-col">
                  <span title={`${r.description}\n\n${r.task || ""}`}
                        className="cursor-default truncate font-mono text-[10px] leading-snug text-slate-300">
                    {r.description}
                  </span>
                  <span className="font-mono text-[9px] text-slate-600">
                    {schedLabel(r)} · {r.enabled ? `next in ${nextIn(r.next_run)}` : "paused"}
                    {` · ${r.runs} run${r.runs === 1 ? "" : "s"}`}
                  </span>
                </div>
                {onRoutineAction && (
                  <div className="flex shrink-0 items-center">
                    <IconBtn icon={Play} title="run now" color="#3ddc97"
                             onClick={() => onRoutineAction("run", r.id)} />
                    <IconBtn icon={r.enabled ? Pause : Play}
                             title={r.enabled ? "pause" : "resume"}
                             onClick={() => onRoutineAction(r.enabled ? "pause" : "resume", r.id)} />
                    <IconBtn icon={Trash2} title="delete" color="#e5484d"
                             onClick={() => onRoutineAction("remove", r.id)} />
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Claude Code bridge: token budget + exchange history */}
      {claude && (
        <div className="hairline rounded-md bg-[#0e0f12]/70 px-3 py-2.5">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Code2 size={12} color="#00d4ff" strokeWidth={1.5} />
              <SectionLabel>CLAUDE</SectionLabel>
            </div>
            {budget?.over && (
              <span className="font-mono text-[9px] tracking-[0.15em] text-[#e5484d]">
                OVER CAP
              </span>
            )}
          </div>

          {/* rolling-24h token meter */}
          <div className="mt-2 flex items-center gap-1.5">
            <Coins size={11} color="#8a94a3" strokeWidth={1.5} />
            <span className="font-mono text-[9px] text-slate-500">
              {fmt(budget?.day)}{budget?.cap ? ` / ${fmt(budget.cap)}` : ""} tok · 24h
            </span>
          </div>
          {budget?.cap > 0 && (
            <div className="mt-1.5 h-1 w-full overflow-hidden rounded-full bg-white/5">
              <div className="h-full rounded-full transition-all"
                   style={{ width: `${Math.round(frac * 100)}%`, background: barColor,
                            boxShadow: `0 0 6px ${barColor}` }} />
            </div>
          )}

          {/* recent exchanges */}
          <div className="mt-2.5 flex flex-col gap-2">
            {exchanges.length === 0 && (
              <span className="font-mono text-[10px] text-slate-600">no exchanges yet</span>
            )}
            {exchanges.map((e) => (
              <div key={e.id} className="flex items-start gap-2"
                   title={`${e.task}\n\n[${e.status}] ${e.tokens || 0} tok${e.detail ? `\n\n${e.detail}` : ""}`}>
                <span className="mt-1 h-1.5 w-1.5 shrink-0 rounded-full"
                      style={{ background: CLAUDE_DOT[e.status] || "#8a94a3" }} />
                <div className="flex min-w-0 flex-1 flex-col">
                  <span className="cursor-default truncate font-mono text-[10px] leading-snug text-slate-300">
                    {e.task}
                  </span>
                  <span className="font-mono text-[9px] text-slate-600">
                    {e.status} · {fmt(e.tokens)} tok
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* recent background jobs */}
      <div className="hairline flex min-h-0 flex-1 flex-col rounded-md bg-[#0e0f12]/70 px-3 py-2.5">
        <SectionLabel>BACKGROUND</SectionLabel>
        <div className="stream mt-2 flex flex-col gap-2 overflow-y-auto">
          {jobs.length === 0 && (
            <span className="font-mono text-[10px] text-slate-600">no jobs yet</span>
          )}
          {jobs.map((j) => {
            const tip = j.task
              + (j.status === "completed" && j.result ? `\n\n→ ${j.result}` : "")
              + (j.status === "failed" && j.error ? `\n\n✗ ${j.error}` : "");
            return (
              <div key={j.id} className="flex items-start gap-2" title={tip}>
                <span className="mt-1 h-1.5 w-1.5 shrink-0 rounded-full"
                      style={{ background: JOB_DOT[j.status] || "#8a94a3",
                               boxShadow: j.status === "running" ? "0 0 6px #00d4ff" : "none" }} />
                <span className="cursor-default font-mono text-[10px] leading-snug text-slate-400">
                  {j.task.length > 54 ? j.task.slice(0, 54) + "…" : j.task}
                </span>
              </div>
            );
          })}
        </div>
      </div>

      {/* footer: location + model + voice */}
      <div className="mt-auto flex flex-col gap-2">
        {where && (
          <span className="flex items-center gap-1.5 font-mono text-[10px] text-slate-500">
            <MapPin size={11} color="#8a94a3" strokeWidth={1.5} />
            <span className="truncate">{where}</span>
          </span>
        )}
        <span className="font-mono text-[9px] tracking-[0.2em] text-slate-600 truncate">
          {status?.model || "—"}
        </span>
        <button
          onClick={() => onToggleVoice(!voiceOn)}
          className={`hairline flex items-center justify-center gap-2 rounded-md py-2 font-mono text-[10px] tracking-[0.2em] transition-colors ${
            voiceOn ? "text-ice glow-ice" : "text-slate-400 hover:text-slate-200"
          }`}
        >
          {voiceOn ? <Volume2 size={13} /> : <VolumeX size={13} />}
          {voiceOn ? "VOICE ON" : "VOICE OFF"}
        </button>
      </div>
    </aside>
  );
}
