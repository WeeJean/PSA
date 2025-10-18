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
  const [response, setResponse] = useState([]);
  const [loading, setLoading] = useState(false);
  const lastMessageRef = useRef(null);
  const askLLM = async () => {
    setLoading(true);
    setQuery("");
    try {
      const res = await fetch("http://127.0.0.1:8000/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });
      const data = await res.json();
      setResponse((prev) => [
        ...prev,
        data.response || data.error || "No response received.",
      ]);
    } catch (err) {
      setResponse("❌ Error connecting to backend.");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);

      try {
        const res = await fetch("http://127.0.0.1:5000/ask", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query: "Give me the summary and actionable insights",
          }),
        });
        const data = await res.json();
        setResponse([
          ...response,
          data.response || data.error || "No response received.",
        ]);
      } catch (err) {
        setResponse("❌ Error connecting to backend.");
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  const dotStyle = (i) => ({
    display: "inline-block",
    animation: `bounce 1.4s infinite ease-in-out ${i * 0.2}s`,
    fontSize: "20px",
    lineHeight: "0",
    padding: "0 4px",
  });

  useEffect(() => {
    lastMessageRef.current?.scrollIntoView({
      behavior: "smooth",
      block: "start",
    });
  }, [response]);

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
              height: "100%", // fills Split pane height
              display: "flex",
              justifyContent: "center", // center card horizontally
              alignItems: "center", // center card vertically (optional)
              padding: "1rem", // space around card
              boxSizing: "border-box",
            }}
          >
            {/* Card container */}
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                width: "100%",
                maxWidth: "500px", // max width of card
                height: "100%", // full height of parent minus padding
                backgroundColor: "#fff",
                borderRadius: "7px",
                boxShadow: "0 4px 10px rgba(0,0,0,0.15)",
                overflow: "hidden", // ensure inner rows stay contained
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
                {response.length > 0 &&
                  response.map((msg, idx) => {
                    const isBot = idx % 2 === 0; // even = bot
                    const isLast = idx === response.length - 1;
                    return (
                      <div
                        key={idx}
                        ref={isLast ? lastMessageRef : null}
                        style={{
                          alignSelf: isBot ? "flex-start" : "flex-end", // left or right
                          backgroundColor: isBot ? "#f0f0f0" : "#1890ff",
                          color: isBot ? "#000" : "#fff",
                          padding: "0.5rem 0.75rem",
                          borderRadius: "12px",
                          maxWidth: "80%",
                          wordBreak: "break-word",
                          textAlign: "left", // ensures content inside bubble is left-aligned
                          marginBottom: "0.25rem",
                        }}
                      >
                        {msg}
                      </div>
                    );
                  })}
                {loading && (
                  <div
                    style={{
                      alignSelf: "flex-start", // left bubble
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
                  placeholder="Type your question..."
                  style={{ flex: 1, resize: "none", fontSize: "14px" }}
                />
                <Button
                  type="primary"
                  onClick={() => {
                    setResponse((prev) => [...prev, query]); // add user message;
                    askLLM();
                  }}
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
