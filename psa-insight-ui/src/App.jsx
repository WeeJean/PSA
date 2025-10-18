import React, { useState } from "react";
import { Input, Button, Card, Typography, Space } from "antd";
import PowerBIReport from "./PowerBIReport";

const { TextArea } = Input;
const { Text, Title } = Typography;

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
        flexDirection: "column", // stack rows vertically
        height: "100vh",
        width: "100vw",
        background: "linear-gradient(180deg, #f0f2f5 0%, #eaf1f8 100%)",
      }}
    >
      {/* 🔹 Row 1: Title + Content */}
      <div
        style={{
          padding: "1rem 2rem",
          textAlign: "center",
          backgroundColor: "#fff",
          boxShadow: "0 2px 4px rgba(0,0,0,0.1)",
        }}
      >
        <Title level={2} style={{ marginBottom: "0.5rem" }}>
          PSA PortSense Dashboard
        </Title>
        <Text type="secondary">
          Monitor port performance and get instant insights from your Copilot.
        </Text>
      </div>

      {/* 🔹 Row 2: Main content area */}
      <div
        style={{
          flex: 1, // takes up remaining space
          display: "flex",
          padding: "1rem",
          gap: "1rem",
          minHeight: 0,
        }}
      >
        {/* Left: Power BI Dashboard */}
        <div style={{ flex: 3, height: "100%" }}>
          <PowerBIReport />
        </div>

        {/* Right: Chat Copilot */}
        <div
          style={{
            flex: 1,
            display: "flex",
            flexDirection: "column",
            height: "100%", // make this side stretch vertically
          }}
        >
          <Card
            title="Insight Copilot"
            style={{
              flex: 1, // make the card fill the full height
              width: "100%",
              maxWidth: 400,
              margin: "0 auto", // center horizontally
              boxShadow: "0 4px 12px rgba(0,0,0,0.1)",
              borderRadius: "8px",
              display: "flex",
              flexDirection: "column", // make contents flow vertically
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
      </div>
    </div>
  );
}
