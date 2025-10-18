import React, { useState } from "react";
import { Input, Button, Card, Typography, Space, Collapse } from "antd";
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
        flexDirection: "column",
        height: "100vh",
        width: "100vw",
        backgroundColor: "#f0f2f5",
        boxSizing: "border-box",
      }}
    >
      {/* Main Content Area */}
      <div
        style={{
          flex: 1,
          display: "flex",
          overflow: "hidden",
        }}
      >
        {/* Left: Collapsible Power BI Dashboard */}
        <div style={{ flex: 3, padding: "1rem" }}>
          <Collapse
            defaultActiveKey={["1"]}
            style={{
              width: "100%",
              background: "#fff",
              borderRadius: "8px",
              boxShadow: "0 2px 8px rgba(0,0,0,0.1)",
            }}
            items={[
              {
                key: "1",
                label: <b>📊 Power BI Dashboard</b>,
                children: (
                  <div
                    style={{
                      height: "85vh",
                      width: "100%",
                      borderRadius: "8px",
                      overflow: "hidden",
                    }}
                  >
                    <PowerBIReport />
                  </div>
                ),
              },
            ]}
          />
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

      {/* Footer */}
      <footer
        style={{
          backgroundColor: "#000",
          color: "#fff",
          textAlign: "center",
          padding: "0.75rem",
          fontWeight: "500",
          fontSize: "14px",
        }}
      >
        CS2101 not gonna be deleted
      </footer>
    </div>
  );
}
