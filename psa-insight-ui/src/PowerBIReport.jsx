import { useEffect, useRef, useState } from "react";
import * as pbi from "powerbi-client";

export default function PowerBIReport() {
  const reportRef = useRef(null);
  const [embedConfig, setEmbedConfig] = useState(null);

  // Create a Power BI service instance once
  const powerbiServiceRef = useRef(
    new pbi.service.Service(
      pbi.factories.hpmFactory,
      pbi.factories.wpmpFactory,
      pbi.factories.routerFactory
    )
  );

  useEffect(() => {
    async function loadReport() {
      try {
        const res = await fetch("http://localhost:8000/get-embed-token");
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

    const powerbiService = powerbiServiceRef.current;

    // Safely clear any existing embed
    powerbiService.reset(reportRef.current);

    const embedConfigObj = {
      type: "report",
      id: embedConfig.reportId,
      embedUrl: embedConfig.embedUrl,
      accessToken: embedConfig.accessToken,
      tokenType: pbi.models.TokenType.Embed,
      permissions: pbi.models.Permissions.All,
      settings: {
        panes: {
          filters: { visible: false },
          pageNavigation: { visible: true },
        },
        background: pbi.models.BackgroundType.Transparent,
      },
    };

    // Embed the report
    powerbiService.embed(reportRef.current, embedConfigObj);
  }, [embedConfig]);

  return (
    <div
      ref={reportRef}
      style={{
        height: "100%",
        width: "100%",
        borderRadius: "10px",
        overflow: "hidden",
        backgroundColor: "#fff",
      }}
    ></div>
  );
}
