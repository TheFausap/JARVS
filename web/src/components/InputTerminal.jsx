import { useState, useRef } from "react";
import { motion } from "framer-motion";
import { ChevronRight, Loader2 } from "lucide-react";

// A sleek single-line command bar that glows when active.
export default function InputTerminal({ onSubmit, busy }) {
  const [value, setValue] = useState("");
  const [focused, setFocused] = useState(false);
  const inputRef = useRef(null);

  const submit = (e) => {
    e?.preventDefault();
    const text = value.trim();
    if (!text || busy) return;
    onSubmit(text);
    setValue("");
  };

  return (
    <form onSubmit={submit} className="px-8 pb-6 pt-2">
      <motion.div
        animate={{
          boxShadow: focused
            ? "0 0 0 1px rgba(0,212,255,0.4), 0 0 22px -6px rgba(0,212,255,0.5)"
            : "0 0 0 1px rgba(27,29,34,1), 0 0 0 0 rgba(0,212,255,0)",
        }}
        transition={{ duration: 0.25 }}
        className="mx-auto flex max-w-3xl items-center gap-3 rounded-lg bg-[#0e0f12]/80 px-4 py-3"
      >
        <ChevronRight size={16} color={focused ? "#00d4ff" : "#4b5563"} strokeWidth={2} />
        <input
          ref={inputRef}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onFocus={() => setFocused(true)}
          onBlur={() => setFocused(false)}
          placeholder={busy ? "JARVS is responding…" : "Enter command…"}
          disabled={busy}
          autoFocus
          className="flex-1 bg-transparent font-mono text-sm text-slate-100 placeholder:text-slate-600 focus:outline-none disabled:opacity-50"
        />
        {busy ? (
          <Loader2 size={15} color="#00d4ff" className="animate-spin" />
        ) : (
          <button
            type="submit"
            className="font-mono text-[10px] tracking-[0.2em] text-slate-500 transition-colors hover:text-ice"
          >
            SEND
          </button>
        )}
      </motion.div>
    </form>
  );
}
