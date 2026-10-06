import { motion } from "framer-motion";

// A minimalist circular element at the heart of the console. It breathes slowly
// when idle and spins its dashed stroke while JARVS is thinking.
const STATE_LABEL = {
  idle: "STANDBY",
  listening: "LISTENING",
  processing: "PROCESSING",
  speaking: "SPEAKING",
  offline: "OFFLINE",
};

export default function StatusRing({ state = "idle" }) {
  const active = state === "processing";
  const offline = state === "offline";
  const ice = "#00d4ff";
  const dim = offline ? "#4b5563" : ice;

  return (
    <div className="relative flex items-center justify-center" style={{ width: 180, height: 180 }}>
      {/* soft halo */}
      <motion.div
        className="absolute rounded-full"
        style={{ width: 180, height: 180, background: `radial-gradient(circle, ${ice}22, transparent 65%)` }}
        animate={{ opacity: offline ? 0.1 : active ? [0.5, 0.9, 0.5] : [0.25, 0.45, 0.25] }}
        transition={{ duration: active ? 1.4 : 4, repeat: Infinity, ease: "easeInOut" }}
      />

      <svg width="180" height="180" viewBox="0 0 180 180" className="absolute">
        {/* static track */}
        <circle cx="90" cy="90" r="76" fill="none" stroke="#1b1d22" strokeWidth="1" />
        {/* breathing inner ring */}
        <motion.circle
          cx="90" cy="90" r="62" fill="none" stroke={dim} strokeWidth="1" strokeOpacity="0.5"
          animate={{ r: offline ? 62 : active ? [62, 64, 62] : [61, 63, 61] }}
          transition={{ duration: active ? 1.2 : 3.5, repeat: Infinity, ease: "easeInOut" }}
        />
        {/* the active arc: a dashed stroke that rotates while processing */}
        <motion.circle
          cx="90" cy="90" r="76" fill="none" stroke={dim} strokeWidth="2"
          strokeLinecap="round"
          strokeDasharray={active ? "40 437" : "2 10"}
          style={{ transformOrigin: "90px 90px", filter: `drop-shadow(0 0 6px ${dim}aa)` }}
          animate={{ rotate: active ? 360 : 0, strokeOpacity: active ? 1 : 0.35 }}
          transition={active
            ? { rotate: { duration: 1.1, repeat: Infinity, ease: "linear" } }
            : { duration: 0.6 }}
        />
      </svg>

      <div className="relative z-10 flex flex-col items-center gap-1 text-center">
        <span className="font-mono text-2xl tracking-[0.3em] text-ice select-none">JARVS</span>
        <motion.span
          key={state}
          className="font-mono text-[10px] tracking-[0.35em]"
          style={{ color: offline ? "#4b5563" : "#8a94a3" }}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          {STATE_LABEL[state] || "STANDBY"}
        </motion.span>
      </div>
    </div>
  );
}
