import React, { useState, useEffect, useRef } from "react";
import Split from "react-split";
import { Input, Button, Space } from "antd";
import PowerBIReport from "./PowerBIReport";
import ReactMarkdown from "react-markdown";

const { TextArea } = Input;

export default function App() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [messages, setMessages] = useState([]);
  const [suggestions, setSuggestions] = useState([]);
  const [powerBIConfig, setPowerBIConfig] = useState(null); // friend’s addition
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [overlayStyle, setOverlayStyle] = useState(null);

  const lastMessageRef = useRef(null);
  const chatRef = useRef(null);
  const hasRun = useRef(false);

  const API_BASE = "http://127.0.0.1:8000";

  // -------------------- askLLM --------------------
  const askLLM = async (forcedQuestion) => {
    const q = (forcedQuestion ?? query).trim();
    if (!q) return;

    setLoading(true);
    setMessages((prev) => [...prev, { role: "user", text: q }]);

    try {
      const res = await fetch(`${API_BASE}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q }),
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `HTTP ${res.status}`);
      }

      const data = await res.json();

      // ----- normalize different backend responses -----
      let assistantText = "";
      let nextSuggestions = [];
      let pbi = null;

      if (data?.mode === "pipeline") {
        assistantText = data.text ?? "";
        nextSuggestions = data.details?.suggestions || data.details?.next_steps || [];
        pbi = data.details?.powerBI ?? null;
      } else if (data?.mode === "agent") {
        assistantText = data.text ?? "";
      } else if (data?.answer_type) {
        assistantText = data.message ?? "";
        nextSuggestions = data.payload?.suggestions ?? [];
        pbi = data.payload?.powerBI ?? null;
      } else {
        assistantText = typeof data === "string" ? data : JSON.stringify(data);
      }

      setMessages((prev) => [...prev, { role: "assistant", text: assistantText }]);
      setSuggestions(nextSuggestions);
      setPowerBIConfig(pbi);
    } catch (err) {
      setMessages((prev) => [...prev, { role: "assistant", text: `⚠️ ${err.message}` }]);
    } finally {
      setLoading(false);
      setQuery("");
    }
  };

  // -------------------- Auto starter question --------------------
  useEffect(() => {
    if (hasRun.current) return;
    hasRun.current = true;

    (async () => {
      try {
        const res = await fetch(`${API_BASE}/ask`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: "Give me the summary of KPI changes and updates, as well as and actionable insights to improve KPI in terms of efficiency and environmental sustainability." }),
        });
        const data = await res.json();

        let assistantText = "";
        let nextSuggestions = [];
        let pbi = null;

        if (data?.mode === "pipeline") {
          assistantText = data.text ?? "";
          nextSuggestions = data.details?.suggestions || data.details?.next_steps || [];
          pbi = data.details?.powerBI ?? null;
        } else if (data?.mode === "agent") {
          assistantText = data.text ?? "";
        } else if (data?.answer_type) {
          assistantText = data.message ?? "";
          nextSuggestions = data.payload?.suggestions ?? [];
          pbi = data.payload?.powerBI ?? null;
        } else {
          assistantText = typeof data === "string" ? data : JSON.stringify(data);
        }

        setMessages((prev) => [...prev, { role: "assistant", text: assistantText }]);
        setSuggestions(nextSuggestions);
        setPowerBIConfig(pbi);
      } catch (err) {
        setMessages((prev) => [...prev, { role: "assistant", text: `⚠️ ${err.message}` }]);
      }
    })();
  }, []);

  // -------------------- Auto scroll --------------------
  useEffect(() => {
    lastMessageRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [messages]);

  // -------------------- Expand/Collapse Chat --------------------
  const expandChat = () => {
    if (!chatRef.current) return;
    const rect = chatRef.current.getBoundingClientRect();

    setOverlayStyle({
      position: "fixed",
      top: rect.top,
      left: rect.left,
      width: rect.width,
      height: rect.height,
      backgroundColor: "#fff",
      zIndex: 9999,
      borderRadius: "7px",
      boxShadow: "0 4px 10px rgba(0,0,0,0.15)",
      overflow: "hidden",
      transition: "all 0.3s ease-in-out",
    });

    setIsFullscreen(true);

    requestAnimationFrame(() => {
      setOverlayStyle((prev) => ({
        ...prev,
        top: 0,
        left: 0,
        width: window.innerWidth,
        height: window.innerHeight,
        borderRadius: 0,
        boxShadow: "none",
      }));
    });
  };

  const collapseChat = () => {
    if (!chatRef.current) return;
    const rect = chatRef.current.getBoundingClientRect();

    setOverlayStyle((prev) => ({
      ...prev,
      top: rect.top,
      left: rect.left,
      width: rect.width,
      height: rect.height,
      borderRadius: "7px",
      boxShadow: "0 4px 10px rgba(0,0,0,0.15)",
    }));

    setTimeout(() => setIsFullscreen(false), 300);
  };

  // -------------------- Chat Content --------------------
  const ChatContent = () => (
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
          backgroundColor: "#fafafa",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
          <img src="./public/BoMen.png" width="30px" alt="Bo-men" />
          <span style={{ fontWeight: "bold", color: "black" }}>Ask Bo-men</span>
        </div>
        <Button
          type="text"
          onClick={isFullscreen ? collapseChat : expandChat}
          style={{ fontSize: "1.25rem", padding: 0, color: "#333", width: "30px" }}
        >
          {isFullscreen ? "✕" : "⛶"}
        </Button>
      </div>

      {/* Messages */}
      <div
        style={{
          flex: 1,
          overflowY: "auto",
          padding: "1rem",
          display: "flex",
          flexDirection: "column",
          gap: "10px",
          fontSize: "14px",
          wordBreak: "break-word",
          color: "black",
          whiteSpace: "pre-wrap",
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
                textAlign: "left",
              }}
            >
              {isBot ? <ReactMarkdown>{m.text}</ReactMarkdown> : m.text}
              {isBot && isLast && suggestions.length > 0 && (
                <div style={{ marginTop: "8px" }}>
                  <Space wrap>
                    {suggestions.map((s, i) => (
                      <Button key={i} size="small" onClick={() => askLLM(s)}>
                        {s}
                      </Button>
                    ))}
                  </Space>
                </div>
              )}
            </div>
          );
        })}
        {loading && <div>Loading...</div>}
      </div>

      {/* Footer */}
      <div
        style={{
          flexShrink: 0,
          display: "flex",
          gap: "0.5rem",
          padding: "0.5rem 1rem",
          borderTop: "1px solid #e0e0e0",
        }}
      >
        <TextArea
          rows={isFullscreen ? 4 : 1}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onPressEnter={(e) => {
            if (!e.shiftKey) {
              e.preventDefault();
              askLLM();
            }
          }}
          placeholder="Type your question..."
          style={{ flex: 1, resize: "none", fontSize: "14px" }}
        />
        <Button
          type="primary"
          onClick={() => askLLM()}
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            height: "100%",
            width: isFullscreen ? "60px" : "auto",
          }}
        >
          ➤
        </Button>
      </div>
    </div>
  );

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100vh", width: "100vw", backgroundColor: "#f0f2f5" }}>
      {/* HEADER */}
      <header
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          gap: "1rem",
          padding: "1rem 2rem",
          backgroundColor: "#fff",
          boxShadow: "0 2px 4px rgba(0,0,0,0.1)",
        }}
      >
        <div style={{ textAlign: "center" }}>
          <h2 style={{ margin: 0, color: "black" }}>PSA PortSense Dashboard</h2>
          <span style={{ color: "#666" }}>
            Monitor port performance and get instant insights from your Copilot.
          </span>
        </div>
      </header>

      {/* MAIN */}
      <main style={{ flex: 1, display: "flex", overflow: "hidden" }}>
        <Split sizes={[70, 30]} minSize={0} gutterSize={5} style={{ display: "flex", flex: 1 }}>
          {/* Left */}
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

          {/* Right */}
          <div style={{ height: "100%", padding: "1rem", display: "flex", justifyContent: "center", alignItems: "center" }}>
            <div ref={chatRef} style={{ width: "100%", height: "100%" }}>
              <ChatContent />
            </div>
          </div>
        </Split>
      </main>

      {/* Footer */}
      <footer style={{ backgroundColor: "#2b2b2b", color: "#fff", textAlign: "center", padding: "0.5rem" }}>
        CS2101 not gonna be deleted
      </footer>

      {/* Fullscreen Overlay */}
      {isFullscreen && overlayStyle && (
        <div style={{ ...overlayStyle, display: "flex", flexDirection: "column" }}>
          <ChatContent />
        </div>
      )}
    </div>
  );
}
