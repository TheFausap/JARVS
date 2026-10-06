import { motion } from "framer-motion";
import {
  Activity, Cpu, Database, Network, BrainCircuit, Volume2, VolumeX, Radio, ListChecks,
} from "lucide-react";

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
        <span className="font-mono text-[9px] tracking-[0.22em] text-slate-500">
          {label}
        </span>
      </div>
      <div className="mt-1.5 flex items-baseline gap-1">
        <span
          className="font-mono text-lg"
          style={{ color: accent ? "#00d4ff" : "#d7dbe3" }}
        >
          {value}
        </span>
        {unit && <span className="font-mono text-[10px] text-slate-600">{unit}</span>}
      </div>
    </motion.div>
  );
}

export default function DataDashboard({ stats, status, jobs = [], onToggleVoice }) {
  const online = status?.backend_ok;
  const voiceOn = stats?.voice_enabled ?? status?.voice_enabled;
  const jobCounts = stats?.jobs || {};
  const activeJobs = (jobCounts.queued || 0) + (jobCounts.running || 0);

  return (
    <aside className="hairline flex w-72 shrink-0 flex-col gap-4 border-y-0 border-r-0 bg-[#0b0c0e]/60 p-5 backdrop-blur-sm">
      {/* system heartbeat */}
      <div className="flex items-center justify-between">
        <span className="font-mono text-[10px] tracking-[0.3em] text-slate-400">
          SYSTEM
        </span>
        <span className="flex items-center gap-1.5 font-mono text-[10px]">
          <Radio size={11} color={online ? "#00d4ff" : "#e5484d"} strokeWidth={2} />
          <span style={{ color: online ? "#00d4ff" : "#e5484d" }}>
            {online ? "ONLINE" : "OFFLINE"}
          </span>
        </span>
      </div>

      <div className="grid grid-cols-2 gap-2.5">
        <Widget icon={Activity} label="LATENCY" value={stats?.latency_ms ?? "—"} unit="ms" accent />
        <Widget icon={Cpu} label="TOKENS" value={stats?.last_tokens ?? "—"} />
        <Widget icon={Database} label="EPISODIC" value={stats?.episodic ?? "—"} />
        <Widget icon={Database} label="SEMANTIC" value={stats?.semantic ?? "—"} />
        <Widget icon={Network} label="LINKS" value={stats?.links ?? "—"} />
        <Widget icon={BrainCircuit} label="FOCUS" value={stats?.focus_promoted ?? "—"} accent />
        <Widget icon={ListChecks} label="JOBS" value={activeJobs} accent={activeJobs > 0} />
      </div>

      {/* recent background jobs */}
      <div className="hairline flex min-h-0 flex-1 flex-col rounded-md bg-[#0e0f12]/70 px-3 py-2.5">
        <span className="font-mono text-[9px] tracking-[0.22em] text-slate-500">BACKGROUND</span>
        <div className="stream mt-2 flex flex-col gap-2 overflow-y-auto">
          {jobs.length === 0 && (
            <span className="font-mono text-[10px] text-slate-600">no jobs yet</span>
          )}
          {jobs.map((j) => (
            <div key={j.id} className="flex items-start gap-2">
              <span className="mt-1 h-1.5 w-1.5 shrink-0 rounded-full"
                    style={{ background: JOB_DOT[j.status] || "#8a94a3",
                             boxShadow: j.status === "running" ? "0 0 6px #00d4ff" : "none" }} />
              <span className="font-mono text-[10px] leading-snug text-slate-400">
                {j.task.length > 54 ? j.task.slice(0, 54) + "…" : j.task}
              </span>
            </div>
          ))}
        </div>
      </div>

      <div className="hairline rounded-md bg-[#0e0f12]/70 px-3 py-2">
        <span className="font-mono text-[9px] tracking-[0.22em] text-slate-500">TURNS</span>
        <span className="ml-2 font-mono text-sm text-slate-300">{stats?.turns ?? 0}</span>
      </div>

      <div className="mt-auto flex flex-col gap-2">
        <span className="font-mono text-[9px] tracking-[0.25em] text-slate-600 truncate">
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
