import React, { useState } from "react";
import { Input, Button, Card, Typography, Space } from "antd";
import PowerBIReport from "./PowerBIReport"; // 👈 import it

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
        height: "100vh",
        width: "100vw",
        backgroundColor: "#f0f2f5",
        boxSizing: "border-box",
      }}
    >
      {/* Left: Power BI Dashboard */}
      <div style={{ flex: 3, padding: "1rem" }}>
        <PowerBIReport />
      </div>

      {/* Right: Chat Copilot */}
      <div
        style={{
          flex: 1,
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
          padding: "1rem",
        }}
      >
        <Card
          title="PSA Insight Copilot"
          style={{
            width: "100%",
            maxWidth: 400,
            boxShadow: "0 4px 12px rgba(0,0,0,0.1)",
            borderRadius: "8px",
          }}
        >
          <Space direction="vertical" style={{ width: "100%" }}>
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
                  marginTop: "1rem",
                  background: "#fafafa",
                  padding: "1rem",
                  borderRadius: "6px",
                  textAlign: "left",
                  wordBreak: "break-word",
                  maxHeight: "50vh",
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
    </div>
  );
}
