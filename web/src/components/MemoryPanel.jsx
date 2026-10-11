import { motion } from "framer-motion";
import { Database, Layers, Share2, BrainCircuit, Sparkles } from "lucide-react";

function Stat({ icon: Icon, label, value, accent }) {
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
      <div className="mt-1.5 font-mono text-lg" style={{ color: accent ? "#00d4ff" : "#d7dbe3" }}>
        {value}
      </div>
    </motion.div>
  );
}

const dash = (v) => (v === null || v === undefined ? "—" : v);

export default function MemoryPanel({ stats, status }) {
  const mem = status?.memory || {};
  const embed = mem.enabled ? (status?.memory?.dim ? `${status.memory.dim}-dim` : "on") : "off";

  return (
    <aside className="hairline flex w-64 shrink-0 flex-col gap-4 border-y-0 border-l-0 bg-[#0b0c0e]/60 p-5 backdrop-blur-sm">
      <div className="flex items-center justify-between">
        <span className="font-mono text-[10px] tracking-[0.3em] text-slate-400">MEMORY</span>
        <span className="font-mono text-[9px] tracking-[0.2em] text-slate-600">{embed}</span>
      </div>

      <div className="grid grid-cols-2 gap-2.5">
        <Stat icon={Database} label="EPISODIC" value={dash(stats?.episodic)} accent />
        <Stat icon={Layers} label="SEMANTIC" value={dash(stats?.semantic)} />
        <Stat icon={Share2} label="LINKS" value={dash(stats?.links)} />
      </div>

      <div className="flex flex-col gap-2">
        <span className="font-mono text-[9px] tracking-[0.3em] text-slate-600">COGNITION</span>
        <div className="grid grid-cols-2 gap-2.5">
          <Stat icon={BrainCircuit} label="FOCUS" value={dash(stats?.focus_promoted)} accent />
          <Stat icon={Sparkles} label="PENDING" value={dash(stats?.focus_pending)} />
        </div>
      </div>

      <p className="mt-auto font-mono text-[9px] leading-relaxed text-slate-600">
        Episodic = conversation turns · Semantic = distilled facts · Links = self-drawn
        associations · Focus = promoted critique areas.
      </p>
    </aside>
  );
}
