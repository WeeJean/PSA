import { useEffect, useRef, useState } from "react";
import { models, powerbi } from "powerbi-client"; // ✅ import powerbi directly

export default function PowerBIReport() {
  const reportRef = useRef(null);
  const [embedConfig, setEmbedConfig] = useState(null);

  useEffect(() => {
    async function loadReport() {
      try {
        const res = await fetch("http://localhost:5000/get-embed-token");
        const data = await res.json();
        setEmbedConfig(data);
      } catch (err) {
        console.error("Failed to fetch embed config:", err);
      }
    }
    loadReport();
  }, []);

  useEffect(() => {
    if (!embedConfig || !reportRef.current) return;

    // Clear any previous embed (to prevent memory leaks)
    powerbi.reset(reportRef.current);

    const embedConfigObj = {
      type: "report",
      id: embedConfig.reportId,
      embedUrl: embedConfig.embedUrl,
      accessToken: embedConfig.accessToken,
      tokenType: models.TokenType.Embed,
      permissions: models.Permissions.All,
      settings: {
        panes: {
          filters: { visible: false },
          pageNavigation: { visible: true },
        },
        background: models.BackgroundType.Transparent,
      },
    };

    // ✅ Embed the report
    powerbi.embed(reportRef.current, embedConfigObj);
  }, [embedConfig]);

  return (
    <div
      ref={reportRef}
      style={{
        height: "90vh",
        width: "100%",
        borderRadius: "10px",
        overflow: "hidden",
        backgroundColor: "#fff",
      }}
    ></div>
  );
}
