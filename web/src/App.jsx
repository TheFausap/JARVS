import { useEffect, useRef, useState } from "react";
import StatusRing from "./components/StatusRing.jsx";
import MessageStream from "./components/MessageStream.jsx";
import DataDashboard from "./components/DataDashboard.jsx";
import InputTerminal from "./components/InputTerminal.jsx";
import { getStatus, getStats, getJobs, setVoice, streamChat } from "./lib/api.js";

let _id = 0;
const nextId = () => `m${_id++}`;

export default function App() {
  const [messages, setMessages] = useState([]);
  const [state, setState] = useState("idle"); // idle|listening|processing|speaking|offline
  const [busy, setBusy] = useState(false);
  const [stats, setStats] = useState(null);
  const [status, setStatus] = useState(null);
  const [jobs, setJobs] = useState([]);
  const busyRef = useRef(false);
  const seenDoneRef = useRef(new Set());
  const firstJobsRef = useRef(true);

  // Initial status + periodic dashboard / jobs polling.
  useEffect(() => {
    let alive = true;
    getStatus().then((s) => alive && setStatus(s)).catch(() => {});
    const tick = () => {
      getStats()
        .then((s) => {
          if (!alive) return;
          setStats(s);
          if (!busyRef.current) setState(s.backend_ok ? "idle" : "offline");
        })
        .catch(() => alive && !busyRef.current && setState("offline"));
      getJobs()
        .then((d) => {
          if (!alive || !d.enabled) return;
          setJobs(d.jobs || []);
          // Surface newly completed jobs into the transcript as background replies.
          for (const j of d.jobs || []) {
            if (j.status === "completed" && !seenDoneRef.current.has(j.id)) {
              seenDoneRef.current.add(j.id);
              if (!firstJobsRef.current && j.result) {
                setMessages((m) => [...m, {
                  id: nextId(), role: "assistant", background: true,
                  text: `⟡ Background task — ${j.task}\n\n${j.result}`,
                }]);
              }
            }
          }
          firstJobsRef.current = false;  // don't replay history on first load
        })
        .catch(() => {});
    };
    tick();
    const h = setInterval(tick, 3000);
    return () => { alive = false; clearInterval(h); };
  }, []);

  const toggleVoice = async (enabled) => {
    try {
      const r = await setVoice(enabled);
      setStatus((s) => ({ ...s, ...r }));
      setStats((s) => (s ? { ...s, voice_enabled: r.voice_enabled } : s));
    } catch { /* ignore */ }
  };

  const send = async (text) => {
    const userMsg = { id: nextId(), role: "user", text };
    const botId = nextId();
    setMessages((m) => [...m, userMsg, { id: botId, role: "assistant", text: "", streaming: true }]);
    setBusy(true); busyRef.current = true;
    setState("processing");

    const append = (delta) =>
      setMessages((m) => m.map((x) => (x.id === botId ? { ...x, text: x.text + delta } : x)));
    const finalize = (patch) =>
      setMessages((m) => m.map((x) => (x.id === botId ? { ...x, ...patch, streaming: false } : x)));

    await streamChat(text, {
      onToken: append,
      onDone: ({ reply }) => {
        finalize(reply ? { text: reply } : {});
        // brief "speaking" beat if voice is on, else back to idle
        if (status?.voice_enabled || stats?.voice_enabled) {
          setState("speaking");
          setTimeout(() => !busyRef.current && setState("idle"), 1200);
        } else {
          setState("idle");
        }
      },
      onError: (detail) => finalize({ text: `⚠ ${detail}` }),
    });

    setBusy(false); busyRef.current = false;
    setState((s) => (s === "speaking" ? s : "idle"));
  };

  return (
    <div className="flex h-full w-full">
      <DataDashboard stats={stats} status={status} jobs={jobs} onToggleVoice={toggleVoice} />

      <main className="flex min-w-0 flex-1 flex-col">
        {/* header: the central status ring */}
        <header className="flex items-center justify-center pt-8 pb-2">
          <StatusRing state={state} />
        </header>

        <MessageStream messages={messages} />
        <InputTerminal onSubmit={send} busy={busy} />
      </main>
    </div>
  );
}
