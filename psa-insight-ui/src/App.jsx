<<<<<<< HEAD
import React, { useState, useEffect, useRef, useLayoutEffect } from "react";
=======
import { useState, useEffect, useRef } from "react";
>>>>>>> e859688299f25624484da3bd65f96d815f173094
import Split from "react-split";
import { Input, Button, Card, Typography, Space } from "antd";
import PowerBIReport from "./PowerBIReport";
import ReactMarkdown from "react-markdown";

const { TextArea } = Input;

export default function App() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [powerBIConfig, setPowerBIConfig] = useState(null); // optional: if agent returns embed
  const lastMessageRef = useRef(null);
  const [messages, setMessages] = useState([]); // [{role:'user'|'assistant', text:string}]
  const [suggestions, setSuggestions] = useState([]); // pipeline next steps
<<<<<<< HEAD
  const chipRowRef = useRef(null);
  const [chipRowH, setChipRowH] = useState(40); // reserve space under textarea
=======
  const hasRun = useRef(false);
>>>>>>> e859688299f25624484da3bd65f96d815f173094

  const API_BASE = "http://127.0.0.1:8000";

  useLayoutEffect(() => {
    if (chipRowRef.current) {
      const h = chipRowRef.current.getBoundingClientRect().height || 40;
      setChipRowH(h);
    }
  }, [suggestions]);

  function extractSuggestionsFromText(s, max = 5) {
    if (!s) return [];
    const lines = String(s)
      .split(/\r?\n/)
      .map((l) => l.trim())
      .filter(Boolean);
    const candidates = lines.filter(
      (l) =>
        /^[-*]\s+/.test(l) || // bullets
        /^\d+\.\s+/.test(l) || // numbered
        /^(increase|reduce|review|investigate|optimi[sz]e|monitor|coordinate|escalate|deploy|pilot|train|audit|fix|patch|tune|rebalance|re[- ]?route|communicate|benchmark|validate|improve|lower|boost|align)\b/i.test(
          l
        ) // imperative-ish
    );
    const cleaned = candidates.map((l) => l.replace(/^[-*\d.]+\s+/, "").trim());
    const out = [],
      seen = new Set();
    for (const c of cleaned) {
      const k = c.toLowerCase();
      if (!seen.has(k) && c) {
        seen.add(k);
        out.push(c.slice(0, 120));
        if (out.length >= max) break;
      }
    }
    return out;
  }

  const askLLM = async (forcedQuestion) => {
    const q = (forcedQuestion ?? query).trim();
    if (!q) return;

    setLoading(true);
    setMessages((prev) => [...prev, { role: "user", text: q }]);

    try {
      const res = await fetch(`${API_BASE}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q }), // no mode, agent-only backend
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
        data?.message ?? // legacy safety
        (typeof data === "string" ? data : JSON.stringify(data));

      let chips = data?.details?.suggestions ?? [];
      if (!chips.length) {
        // derive chips from the text if backend didn't provide any
        chips = extractSuggestionsFromText(assistantText, 5);
      }

      const pbi =
        data?.details?.powerBI ??
        data?.payload?.powerBI ?? // legacy safety
        null;

      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: assistantText },
      ]);
      setSuggestions(chips);
      if (pbi) setPowerBIConfig(pbi);
    } catch (err) {
      console.error(err);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: `⚠️ ${err.message || "Request failed"}` },
      ]);
    } finally {
      setLoading(false);
      setQuery(""); // clear AFTER sending
    }
  };

<<<<<<< HEAD
  const SuggestionChips = ({ items, onClick }) => {
=======
  console.log(messages);
  const SuggestionChips = ({ items }) => {
>>>>>>> e859688299f25624484da3bd65f96d815f173094
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
            onClick={() => onClick?.(s)}
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
        // Normalize the 3 possible shapes
        let assistantText = "";
        let nextSuggestions = [];
        let pbi = null;
        const data = await res.json();
        if (data?.mode === "pipeline") {
          // {mode:"pipeline", text, details}
          assistantText = data.text ?? "";
          nextSuggestions =
            data.details?.suggestions || data.details?.next_steps || [];
          pbi = data.details?.powerBI ?? null;
        } else if (data?.mode === "agent") {
          // {mode:"agent", text}
          assistantText = data.text ?? "";
          // agent mode typically has no suggestions
        } else if (data?.answer_type) {
          // Envelope: {answer_type, message, payload}
          assistantText = data.message ?? "";
          nextSuggestions = data.payload?.suggestions ?? [];
          pbi = data.payload?.powerBI ?? null;
        } else {
          // Fallback: show whatever we got (useful while integrating)
          assistantText =
            typeof data === "string" ? data : JSON.stringify(data);
        }
        setMessages((prev) => [
          ...prev,
          { role: "assistant", text: assistantText },
        ]);
        setSuggestions(nextSuggestions);
        setPowerBIConfig(pbi);
      } catch (err) {
        setMessages((prev) => [
          ...prev,
          { role: "assistant", text: `⚠️ ${err.message}` },
        ]);
      } finally {
        setLoading(false);
        setQuery(""); // clear AFTER sending
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    lastMessageRef.current?.scrollIntoView({
      behavior: "smooth",
      block: "start",
    });
  }, [response]);

  const dotStyle = (i) => ({
    display: "inline-block",
    animation: `bounce 1.4s infinite ease-in-out ${i * 0.2}s`,
    fontSize: "20px",
    lineHeight: "0",
    padding: "0 4px",
  });

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
              src="./BoMen.png"
              style={{ backgroundColor: "white", paddingRight: "4px" }}
              width="30px"
            ></img>
            <h2 style={{ margin: 0, color: "black" }}>
              PSA PortSense Dashboard
            </h2>
          </div>

          <span style={{ color: "#666" }}>
            Monitor port performance and get instant insights from your Copilot.
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
          sizes={[70, 30]} // default split
          minSize={300}
          gutterSize={5}
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
                ‎ ‎ Ask Bo-men
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
                      ref={isLast ? lastMessageRef : null}
                      style={{
                        alignSelf: isBot ? "flex-start" : "flex-end",
                        backgroundColor: isBot ? "#f0f0f0" : "#1890ff",
                        color: isBot ? "#000" : "#fff",
                        padding: "0.75rem 0.9rem",
                        borderRadius: 12,
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

                      {isBot && isLast && suggestions?.length > 0 && (
                        <SuggestionChips
                          items={suggestions}
                          onClick={(s) => askLLM(s)}
                        />
                      )}
                    </div>
                  );
                })}

                {loading && (
                  <div
                    style={{
                      alignSelf: "flex-start",
                      backgroundColor: "#f0f0f0",
                      color: "#000",
                      padding: "0.5rem 0.75rem",
                      borderRadius: "12px",
                      maxWidth: "30px",
                      display: "flex",
                      gap: "3px",
                      justifyContent: "center",
                      marginBottom: "0.25rem",
                    }}
                  >
                    <span style={dotStyle(0)}>•</span>
                    <span style={dotStyle(1)}>•</span>
                    <span style={dotStyle(2)}>•</span>
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
                  onClick={() => askLLM()} // DO NOT manually push messages here
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
