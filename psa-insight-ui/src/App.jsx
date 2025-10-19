import { useState, useEffect, useRef } from "react";
import Split from "react-split";
import { Input, Button } from "antd";
import PowerBIReport from "./PowerBIReport";
import ReactMarkdown from "react-markdown";

const { TextArea } = Input;

export default function App() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(true);
  const [powerBIConfig, setPowerBIConfig] = useState(null); // optional: if agent returns embed
  const lastMessageRef = useRef(null);
  const [messages, setMessages] = useState([]); // [{ role: 'user'|'assistant', text: string, suggestions?: string[] }]
  const hasRun = useRef(false);
  const API_BASE = "http://127.0.0.1:8000";

  const DEFAULT_CHIPS = [
    "Summarize KPI snapshot for APAC",
    "Show WoW trend for ArrivalAccuracy(FinalBTR) in APAC",
    "Find anomalies by BU for ArrivalAccuracy(FinalBTR) in APAC",
  ];

  // Helper functions for suggestion chips
  function filterAllowedSuggestions(items, max = 6) {
    if (!Array.isArray(items)) return [];

    const verb =
      /^(Show|Summarize|Compare|Rank|Investigate|List|Peek|Recommend)\b/i;
    const intent = new RegExp(
      [
        "KPI",
        "snapshot",
        "WoW",
        "trend",
        "week",
        "month",
        "MoM",
        "delta",
        "change",
        "rank",
        "best",
        "worst",
        "top",
        "bottom",
        "anomal",
        "outlier",
        "compare",
        " vs ",
        "driver",
        "drill",
        "distinct",
        "unique",
        "values",
        "list",
        "peek",
        "sample",
        "column",
        "actions",
        "steps",
        "improve",
        "recommend",
      ].join("|"),
      "i"
    );

    const out = [];
    const seen = new Set();

    for (const s of items) {
      if (typeof s !== "string") continue;
      const t = s.trim().replace(/\.$/, "");
      if (!t) continue;
      if (!verb.test(t)) continue;
      if (!intent.test(t)) continue;

      const k = t.toLowerCase();
      if (seen.has(k)) continue;
      seen.add(k);
      out.push(t);
      if (out.length >= max) break;
    }
    return out;
  }

  function getLastAssistantChips(msgs) {
    for (let i = msgs.length - 1; i >= 0; i--) {
      if (msgs[i].role === "assistant" && Array.isArray(msgs[i].suggestions)) {
        return msgs[i].suggestions;
      }
    }
    return [];
  }

  function getLastUserQuestion(msgs) {
    for (let i = msgs.length - 1; i >= 0; i--) {
      if (msgs[i].role === "user") return msgs[i].text || "";
    }
    return "";
  }

  const askLLM = async (forcedQuestion) => {
    const q = (forcedQuestion ?? query).trim();
    if (!q) return;
    setMessages((prev) => [...prev, { role: "user", text: q }]);
    setLoading(true);
    setTimeout(() => {}, 5000);
    setQuery("");

    try {
      const lastChips = getLastAssistantChips(messages);
      const lastQ = getLastUserQuestion(messages);

      const res = await fetch(`${API_BASE}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: q,
          last_question: lastQ,
          recent_suggestions: lastChips,
        }),
      });

      const raw = await res.text();
      if (!res.ok) {
        throw new Error(raw || `HTTP ${res.status}`);
      }

      let data;
      try {
        data = JSON.parse(raw);
      } catch {
        data = { text: raw };
      }

      /// Backend contract: { text, details?: { suggestions?, powerBI? } }
      const assistantText =
        data?.text ??
        data?.message ??
        (typeof data === "string" ? data : JSON.stringify(data));

      const rawBackendChips = data?.details?.suggestions ?? [];
      let chips = filterAllowedSuggestions(rawBackendChips, 5);

      // drop anything equal/super-similar to the current q
      chips = chips.filter((c) => c.toLowerCase() !== q.toLowerCase());

      // if backend gave something but we filtered all, keep first 3 raw
      if (!chips.length && rawBackendChips.length)
        chips = rawBackendChips.slice(0, 3);
      if (!chips.length) chips = DEFAULT_CHIPS;

      const pbi = data?.details?.powerBI ?? data?.payload?.powerBI ?? null;

      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: assistantText, suggestions: chips },
      ]);
      if (pbi) setPowerBIConfig(pbi);
    } catch (err) {
      console.error(err);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: `⚠️ ${err.message || "Request failed"}` },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const SuggestionChips = ({ items, onClick }) => {
    if (!items?.length) return null;

    return (
      <div
        style={{
          marginTop: 8,
          display: "flex",
          flexWrap: "wrap",
          gap: 8,
          width: "100%",
        }}
      >
        {items.map((s, i) => (
          <button
            key={i}
            onClick={() => {
              setQuery("");
              onClick?.(s);
            }}
            title={s}
            style={{
              // pill container
              border: "1px solid #e0e0e0",
              background: "#fff",
              borderRadius: 10, // keeps the rounded “pill” shape even when multi-line
              padding: "6px 12px",
              cursor: "pointer",
              display: "inline-block", // allow natural width + wrapping
              maxWidth: "100%", // never overflow bubble width
              textAlign: "left",
              lineHeight: 1.25,
              whiteSpace: "normal", // ✅ allow wrapping inside the button
            }}
          >
            <span
              style={{
                // ✅ let the text wrap; no ellipsis
                whiteSpace: "normal",
                wordBreak: "break-word", // break long words
                overflowWrap: "anywhere", // Safari/iOS friendly
                color: "#333",
                fontSize: 13,
                display: "inline",
              }}
            >
              {s}
            </span>
          </button>
        ))}
      </div>
    );
  };

  // On mount, ask a starter question (fix: use `question`, not `query`)
  useEffect(() => {
    (async () => {
      if (hasRun.current) return;
      hasRun.current = true;
      try {
        const res = await fetch(`${API_BASE}/ask`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            question: "Give me the summary and actionable insights",
          }),
        });

        const raw = await res.text();
        if (!res.ok) {
          throw new Error(raw || `HTTP ${res.status}`);
        }

        let data;
        try {
          data = JSON.parse(raw);
        } catch {
          data = { text: raw };
        }

        // Backend contract: { text, details?: { suggestions?, powerBI? } }
        const assistantText =
          data?.text ??
          data?.message ??
          (typeof data === "string" ? data : JSON.stringify(data));

        const rawBackendChips = data?.details?.suggestions ?? [];
        console.debug("backend suggestions:", rawBackendChips);
        let chips = filterAllowedSuggestions(rawBackendChips, 5);
        if (!chips.length && rawBackendChips.length) {
          console.warn(
            "All backend suggestions filtered out; showing raw first 3"
          );
          chips = rawBackendChips.slice(0, 3); // soft fallback to what the model sent
        }
        if (!chips.length) chips = DEFAULT_CHIPS; // final fallback

        const pbi =
          data?.details?.powerBI ??
          data?.payload?.powerBI ?? // legacy safety
          null;

        setMessages((prev) => [
          ...prev,
          { role: "assistant", text: assistantText, suggestions: chips },
        ]);

        if (pbi) setPowerBIConfig(pbi);
      } catch (err) {
        console.error(err);
        setMessages((prev) => [
          ...prev,
          { role: "assistant", text: `⚠️ ${err.message || "Request failed"}` },
        ]);
      } finally {
        setLoading(false);
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    lastMessageRef.current?.scrollIntoView({
      behavior: "smooth",
      block: "start",
    });
  }, [messages]);

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        height: "100vh",
        width: "100vw",
        backgroundColor: "#f0f2f5",
        boxSizing: "border-box",
      }}
    >
      {/* 🔹 HEADER */}
      <header
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          gap: "1rem",
          padding: "1rem 2rem",
          backgroundColor: "#ffffff",
          boxShadow: "0 2px 4px rgba(0,0,0,0.1)",
          flexShrink: 0,
          borderBottomLeftRadius: "8px",
          borderBottomRightRadius: "8px",
        }}
      >
        <div style={{ textAlign: "center" }}>
          <div
            style={{ display: "flex", width: "100%", justifyContent: "center" }}
          >
            <img
              src="./PSA.png"
              style={{ backgroundColor: "white", paddingRight: "4px" }}
              width="100px"
            ></img>
            <h2 style={{ margin: 0, paddingLeft: "3px", color: "black" }}>
              PortSense Dashboard
            </h2>
          </div>

          <span style={{ color: "#666" }}>
            Monitor port performance and get instant insights from Boman
          </span>
        </div>
      </header>

      {/* MAIN CONTENT (draggable) */}
      <main
        style={{
          flex: 1,
          overflow: "hidden",
          minHeight: 0,
          display: "flex",
        }}
      >
        <Split
          sizes={[60, 40]} // default split
          minSize={0}
          gutterSize={8}
          cursor="col-resize"
          style={{
            display: "flex",
            flex: 1,
            height: "100%",
            minHeight: 0,
          }}
        >
          {/* Left: Dashboard */}
          <div style={{ padding: "1rem", overflow: "hidden" }}>
            <div
              style={{
                height: "100%",
                background: "#fff",
                borderRadius: "8px",
                boxShadow: "0 2px 8px rgba(0,0,0,0.1)",
                overflow: "hidden",
              }}
            >
              <PowerBIReport />
            </div>
          </div>

          {/* Right: Chat Copilot */}
          <div
            style={{
              height: "100%",
              display: "flex",
              justifyContent: "center",
              alignItems: "center",
              padding: "1rem",
              boxSizing: "border-box",
            }}
          >
            {/* Card container */}
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                width: "100%",
                height: "100%",
                backgroundColor: "#fff",
                borderRadius: "7px",
                boxShadow: "0 4px 10px rgba(0,0,0,0.15)",
                overflow: "hidden",
              }}
            >
              {/* Header */}
              <div
                style={{
                  flexShrink: 0,
                  padding: "0.75rem 1rem",
                  borderBottom: "1px solid #e0e0e0",
                  fontWeight: "bold",
                  color: "black",
                  backgroundColor: "#fafafa",
                }}
              >
                <img src="./public/BoMen.png" width="30px"></img>
                ‎ ‎ AskBoman
                <div
                  style={{
                    width: "12px",
                    height: "12px",
                    borderRadius: "50%",
                    backgroundColor: "#05b714",
                    display: "inline-block",
                    marginRight: "8px",
                    marginLeft: "6px",
                  }}
                />
              </div>

              {/* Response area */}
              <div
                style={{
                  flex: 1,
                  minHeight: 0,
                  overflowY: "auto",
                  width: "100%",
                  padding: "1rem",
                  fontSize: "14px",
                  wordBreak: "break-word",
                  color: "black",
                  whiteSpace: "pre-wrap",
                  display: "flex",
                  flexDirection: "column",
                  gap: "10px",
                }}
              >
                {messages.map((m, idx) => {
                  const isBot = m.role === "assistant";
                  const isLast = idx === messages.length - 1;
                  return (
                    <div
                      key={idx}
                      ref={isLast && !loading ? lastMessageRef : null}
                      style={{
                        alignSelf: isBot ? "flex-start" : "flex-end",
                        backgroundColor: isBot ? "#f0f0f0" : "#1890ff",
                        color: isBot ? "#000" : "#fff",
                        padding: "0.5rem 0.75rem",
                        borderRadius: "12px",
                        maxWidth: "80%",
                        textAlign: "left",
                        marginBottom: 8,
                        display: "flex",
                        flexDirection: "column",
                        gap: 8, // internal spacing between text & chips
                        boxSizing: "border-box",
                      }}
                    >
                      <div
                        style={{
                          // keep markdown text constrained to bubble width
                          overflowWrap: "anywhere",
                          whiteSpace: "pre-wrap",
                          lineHeight: 1.45,
                        }}
                      >
                        {isBot ? (
                          <ReactMarkdown>{m.text}</ReactMarkdown>
                        ) : (
                          m.text
                        )}
                      </div>

                      {isBot && m.suggestions?.length > 0 && (
                        <SuggestionChips
                          items={m.suggestions}
                          onClick={(s) => {
                            askLLM(s);
                          }}
                        />
                      )}
                    </div>
                  );
                })}
                {loading && (
                  <div ref={lastMessageRef}>
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      viewBox="0 0 200 200"
                      width="50"
                      height="50"
                    >
                      <circle
                        fill="%23130208"
                        stroke="%23130208"
                        stroke-width="15"
                        r="15"
                        cx="40"
                        cy="65"
                      >
                        <animate
                          attributeName="cy"
                          calcMode="spline"
                          dur="2"
                          values="65;135;65;"
                          keySplines=".5 0 .5 1;.5 0 .5 1"
                          repeatCount="indefinite"
                          begin="-.4"
                        ></animate>
                      </circle>
                      <circle
                        fill="%23130208"
                        stroke="%23130208"
                        stroke-width="15"
                        r="15"
                        cx="100"
                        cy="65"
                      >
                        <animate
                          attributeName="cy"
                          calcMode="spline"
                          dur="2"
                          values="65;135;65;"
                          keySplines=".5 0 .5 1;.5 0 .5 1"
                          repeatCount="indefinite"
                          begin="-.2"
                        ></animate>
                      </circle>
                      <circle
                        fill="%23130208"
                        stroke="%23130208"
                        stroke-width="15"
                        r="15"
                        cx="160"
                        cy="65"
                      >
                        <animate
                          attributeName="cy"
                          calcMode="spline"
                          dur="2"
                          values="65;135;65;"
                          keySplines=".5 0 .5 1;.5 0 .5 1"
                          repeatCount="indefinite"
                          begin="0"
                        ></animate>
                      </circle>
                    </svg>
                  </div>
                )}
              </div>

              {/* Footer: input + button */}
              <div
                style={{
                  flexShrink: 0,
                  display: "flex",
                  gap: "0.5rem",
                  padding: "0.5rem 1rem",
                  borderTop: "1px solid #e0e0e0",
                  alignItems: "flex-end",
                }}
              >
                <TextArea
                  rows={1}
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onPressEnter={(e) => {
                    if (!e.shiftKey) {
                      e.preventDefault();
                      askLLM(); // handles adding the user message internally
                    }
                  }}
                  placeholder="Type your question..."
                  style={{ flex: 1, resize: "none", fontSize: "14px" }}
                />
                <Button
                  type="primary"
                  onClick={() => {
                    askLLM();
                  }} // DO NOT manually push messages here
                  style={{
                    height: "100%",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                  }}
                >
                  ➤
                </Button>
              </div>
            </div>
          </div>
        </Split>
      </main>

      {/* FOOTER */}
      <footer
        style={{
          backgroundColor: "#2b2b2bff",
          color: "#fff",
          textAlign: "center",
          padding: "0.5rem",
          fontWeight: "500",
          fontSize: "14px",
          flexShrink: 0,
        }}
      >
        CS2101 not gonna be deleted
      </footer>
    </div>
  );
}
