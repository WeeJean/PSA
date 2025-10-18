import React, { useState } from "react";
import Split from "react-split";
import { Input, Button, Card, Typography, Space } from "antd";
import PowerBIReport from "./PowerBIReport";

const { TextArea } = Input;
const { Text } = Typography;

export default function App() {
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState("");
  const [loading, setLoading] = useState(false);

  const askLLM = async () => {
    if (!query.trim()) return;
    setLoading(true);
    setResponse("");

    try {
      const res = await fetch("http://127.0.0.1:5000/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });
      const data = await res.json();
      setResponse(data.response || data.error || "No response received.");
    } catch (err) {
      setResponse("❌ Error connecting to backend.");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

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
              display: "flex",
              justifyContent: "center",
              alignItems: "center",
              padding: "1rem",
              overflow: "hidden",
            }}
          >
            <Card
              title="Insight Copilot"
              style={{
                width: "100%",
                maxWidth: "100%",
                boxShadow: "0 4px 12px rgba(0,0,0,0.1)",
                borderRadius: "8px",
                height: "100%",
                display: "flex",
                flexDirection: "column",
              }}
            >
              <Space direction="vertical" style={{ width: "100%", flex: 1 }}>
                <TextArea
                  rows={3}
                  placeholder="Ask something about port performance..."
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                />
                <Button type="primary" loading={loading} onClick={askLLM}>
                  Ask
                </Button>
                {response && (
                  <div
                    style={{
                      flex: 1,
                      marginTop: "1rem",
                      background: "#fafafa",
                      padding: "1rem",
                      borderRadius: "6px",
                      textAlign: "left",
                      wordBreak: "break-word",
                      overflowY: "auto",
                    }}
                  >
                    <Text strong>Response:</Text>
                    <div>{response}</div>
                  </div>
                )}
              </Space>
            </Card>
          </div>
        </Split>
      </main>

      {/* 🔹 FOOTER */}
      <footer
        style={{
          backgroundColor: "#000",
          color: "#fff",
          textAlign: "center",
          padding: "0.75rem",
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
