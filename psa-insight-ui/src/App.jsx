import React, { useState, useEffect, useRef } from "react";
import Split from "react-split";
import { Input, Button, Card, Typography, Space } from "antd";
import { Scrollbar } from "react-scrollbars-custom";
import PowerBIReport from "./PowerBIReport";
import ReactMarkdown from "react-markdown";

const { TextArea } = Input;
const { Text } = Typography;

export default function App() {
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState([]); // array of strings (alternating user/bot)
  const [loading, setLoading] = useState(false);
  const [powerBIConfig, setPowerBIConfig] = useState(null); // optional: if agent returns embed
  const lastMessageRef = useRef(null);
  const [messages, setMessages] = useState([]); // [{role:'user'|'assistant', text:string}]
  const [suggestions, setSuggestions] = useState([]); // pipeline next steps

  const API_BASE = "http://127.0.0.1:8000";

  const askLLM = async (forcedQuestion) => {
    const q = (forcedQuestion ?? query).trim();
    if (!q) return;

    setLoading(true);

    // Show the user's message immediately
    setMessages((prev) => [...prev, { role: "user", text: q }]);

    try {
      const res = await fetch(`${API_BASE}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        // IMPORTANT: backend expects { question: "<text>" }
        body: JSON.stringify({ question: q }),
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `HTTP ${res.status}`);
      }
      const data = await res.json();

      // Normalize the 3 possible shapes
      let assistantText = "";
      let nextSuggestions = [];
      let pbi = null;

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
        assistantText = typeof data === "string" ? data : JSON.stringify(data);
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
  };

  const SuggestionChips = ({ items }) => {
    if (!items?.length) return null;
    return (
      <Space wrap style={{ marginTop: 8 }}>
        {items.map((s, i) => (
          <Button key={i} size="small" onClick={() => askLLM(s)}>
            {s}
          </Button>
        ))}
      </Space>
    );
  };

  // On mount, ask a starter question (fix: use `question`, not `query`)
  useEffect(() => {
    (async () => {
      try {
        const res = await fetch(`${API_BASE}/ask`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            question: "Give me the summary and actionable insights",
          }),
        });
        const data = await res.json();
        // Reuse the same handlers by simulating a single “bot” message:
        if (data.mode === "pipeline") {
          const pretty = data.details
            ? `\n\n\`\`\`json\n${JSON.stringify(data.details, null, 2)}\n\`\`\``
            : "";
          setResponse((prev) => [...prev, `${data.text || ""}${pretty}`]);
        } else if (data.mode === "agent") {
          setResponse((prev) => [...prev, data.text || "(no text)"]);
        } else if (data.answer_type) {
          // simple render path for initial question
          setResponse((prev) => [...prev, data.message || "(no text)"]);
        } else {
          setResponse((prev) => [
            ...prev,
            data.response || data.error || "No response received.",
          ]);
        }
      } catch (e) {
        setResponse((prev) => [...prev, "❌ Error connecting to backend."]);
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
          <h2 style={{ margin: 0, color: "black" }}>PSA PortSense Dashboard</h2>
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
                maxWidth: "500px",
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
                Insight Copilot
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
                        padding: "0.5rem 0.75rem",
                        borderRadius: "12px",
                        maxWidth: "80%",
                        wordBreak: "break-word",
                        textAlign: "left",
                        marginBottom: "0.25rem",
                      }}
                    >
                      {isBot ? <ReactMarkdown>{m.text}</ReactMarkdown> : m.text}

                      {/* Show suggestion chips only under the last assistant message */}
                      {isBot && isLast && suggestions?.length > 0 && (
                        <div style={{ marginTop: "8px" }}>
                          <Space wrap>
                            {suggestions.map((s, i) => (
                              <Button
                                key={i}
                                size="small"
                                onClick={() => askLLM(s)}
                              >
                                {s}
                              </Button>
                            ))}
                          </Space>
                        </div>
                      )}
                    </div>
                  );
                })}
                {loading && (
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
