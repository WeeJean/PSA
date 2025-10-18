// App.jsx
import React, { useState } from "react";
import { Input, Button, Card, Typography, Space, message, Divider } from "antd";
import PowerBIReport from "./PowerBIReport";

const { TextArea } = Input;
const { Title, Text } = Typography;

const API_BASE = "http://127.0.0.1:8000"; // Flask backend

export default function App() {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);

  // whole agent envelope: { answer_type, message, payload, ... }
  const [resp, setResp] = useState(null);

  // Power BI config if agent chooses to show a report
  const [pbiConfig, setPbiConfig] = useState(null);

  async function submit() {
    if (!question.trim()) return;
    setLoading(true);
    setResp(null);

    try {
      const res = await fetch(`${API_BASE}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          history: [], // optional; wire chat history later if you want
        }),
      });

      const text = await res.text();
      if (!res.ok) throw new Error(`HTTP ${res.status}: ${text.slice(0, 300)}`);

      let data;
      try {
        data = JSON.parse(text);
      } catch {
        data = { answer_type: "text", message: text, payload: {} };
      }

      setResp(data);

      if (data.answer_type === "powerbi" && data.payload) {
        setPbiConfig({
          embedUrl: data.payload.embedUrl,
          reportId: data.payload.reportId,
          accessToken: data.payload.accessToken,
        });
      }
    } catch (err) {
      console.error(err);
      message.error(err.message || "Request failed");
      setResp({
        answer_type: "error",
        message: String(err.message || err),
        payload: {},
      });
    } finally {
      setLoading(false);
    }
  }

  function renderPayload() {
    if (!resp) return null;
    const { answer_type, payload } = resp;

    // Simple table renderer if payload has columns/rows
    if (
      answer_type === "table" &&
      payload &&
      Array.isArray(payload.columns) &&
      Array.isArray(payload.rows)
    ) {
      return (
        <div style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse" }}>
            <thead>
              <tr>
                {payload.columns.map((c) => (
                  <th
                    key={c}
                    style={{
                      textAlign: "left",
                      borderBottom: "1px solid #ddd",
                      padding: "6px",
                    }}
                  >
                    {c}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {payload.rows.map((r, i) => (
                <tr key={i}>
                  {r.map((cell, j) => (
                    <td
                      key={j}
                      style={{
                        borderBottom: "1px solid #f0f0f0",
                        padding: "6px",
                      }}
                    >
                      {String(cell)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
    }

    // Fallback: pretty JSON
    return (
      <pre
        style={{
          background: "#fafafa",
          padding: "12px",
          borderRadius: 6,
          marginTop: 8,
          maxHeight: 300,
          overflow: "auto",
          fontSize: 12,
        }}
      >
        {JSON.stringify(payload ?? {}, null, 2)}
      </pre>
    );
  }

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        height: "100vh",
        width: "100vw",
        background: "linear-gradient(180deg, #f0f2f5 0%, #eaf1f8 100%)",
      }}
    >
      {/* Header */}
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
          Ask anything — the Copilot will choose the right tool automatically.
        </Text>
      </div>

      {/* Main */}
      <div
        style={{
          flex: 1,
          display: "flex",
          padding: "1rem",
          gap: "1rem",
          minHeight: 0,
        }}
      >
        {/* Left: Power BI */}
        <div style={{ flex: 3, height: "100%" }}>
          {/* Pass config only if we have it */}
          <PowerBIReport config={pbiConfig} />
        </div>

        {/* Right: Copilot */}
        <div
          style={{
            flex: 1,
            display: "flex",
            flexDirection: "column",
            height: "100%",
          }}
        >
          <Card
            title="Insight Copilot"
            style={{
              flex: 1,
              width: "100%",
              maxWidth: 420,
              margin: "0 auto",
              boxShadow: "0 4px 12px rgba(0,0,0,0.1)",
              borderRadius: "8px",
              display: "flex",
              flexDirection: "column",
            }}
          >
            <Space direction="vertical" style={{ width: "100%", flex: 1 }}>
              <TextArea
                rows={3}
                placeholder="e.g., Explain APAC performance and actions • or • Show WoW trend for ArrivalAccuracy(FinalBTR) in APAC"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
              />

              <div style={{ display: "flex", gap: 8 }}>
                <Button type="primary" loading={loading} onClick={submit}>
                  Ask
                </Button>
                <Button onClick={() => setQuestion("")} disabled={loading}>
                  Clear
                </Button>
              </div>

              {resp && (
                <>
                  <Divider style={{ margin: "8px 0" }} />
                  <div
                    style={{
                      background: "#fafafa",
                      padding: "12px",
                      borderRadius: "6px",
                      textAlign: "left",
                      wordBreak: "break-word",
                      overflowY: "auto",
                      whiteSpace: "pre-wrap",
                      fontSize: 13,
                      maxHeight: 360,
                    }}
                  >
                    <div style={{ marginBottom: 6 }}>
                      <Text strong>Type:</Text>{" "}
                      <code>{resp.answer_type || "text"}</code>
                    </div>
                    <div style={{ marginBottom: 8 }}>
                      <Text strong>Message:</Text>{" "}
                      {resp.message || "(no message)"}
                    </div>
                    {renderPayload()}
                  </div>
                </>
              )}
            </Space>
          </Card>
        </div>
      </div>
    </div>
  );
}
