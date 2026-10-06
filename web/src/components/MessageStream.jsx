import { useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";

// Clean, bubble-less transcript: user messages left-aligned, JARVS right-aligned,
// separated by hairline dividers and clear typography.
function Message({ role, text, streaming }) {
  const isUser = role === "user";
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
      className={`flex flex-col gap-1 py-4 ${isUser ? "items-start" : "items-end"}`}
    >
      <span
        className="font-mono text-[10px] tracking-[0.25em]"
        style={{ color: isUser ? "#4b5563" : "#00d4ff" }}
      >
        {isUser ? "OPERATOR" : "JARVS"}
      </span>
      <div
        className={`max-w-[80%] leading-relaxed whitespace-pre-wrap text-[15px] ${
          isUser ? "text-left text-slate-200" : "text-right"
        }`}
        style={{ color: isUser ? "#d7dbe3" : "#e9f6fb" }}
      >
        {text}
        {streaming && <span className="caret text-ice">▍</span>}
      </div>
    </motion.div>
  );
}

export default function MessageStream({ messages }) {
  const endRef = useRef(null);
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  return (
    <div className="stream flex-1 overflow-y-auto px-8">
      {messages.length === 0 && (
        <div className="flex h-full items-center justify-center">
          <p className="font-mono text-xs tracking-[0.3em] text-slate-600">
            AWAITING INPUT
          </p>
        </div>
      )}
      <div className="mx-auto max-w-3xl divide-y divide-[#14161b]">
        <AnimatePresence initial={false}>
          {messages.map((m) => (
            <Message key={m.id} role={m.role} text={m.text} streaming={m.streaming} />
          ))}
        </AnimatePresence>
      </div>
      <div ref={endRef} />
    </div>
  );
}
