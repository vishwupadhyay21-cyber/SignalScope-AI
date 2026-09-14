import React, { useEffect, useState } from 'react';
import { ShieldCheck, Radio } from 'lucide-react';
import { checkBackendHealth } from '../services/api';

export const Header: React.FC = () => {
  const [health, setHealth] = useState<{ ok: boolean; message: string }>({
    ok: false,
    message: 'Connecting to Engine...',
  });

  useEffect(() => {
    let isMounted = true;
    const probe = async () => {
      const res = await checkBackendHealth();
      if (isMounted) {
        setHealth(res);
      }
    };
    probe();
    const interval = setInterval(probe, 10000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  return (
    <header className="app-header" role="banner">
      <div className="header-container">
        <div className="brand-section">
          <div className="brand-logo-icon" aria-hidden="true">
            <ShieldCheck size={22} />
          </div>
          <div>
            <div className="brand-title">SignalScope</div>
            <div className="brand-subtitle">AI Media Forensics & Trust Lab</div>
          </div>
        </div>

        <div className="header-meta">
          <div className="sih-badge" title="Smart India Hackathon 2026 - Problem Statement">
            <Radio size={13} aria-hidden="true" />
            <span>SIH 2026 • PS</span>
          </div>

          <div
            className={`health-pill ${health.ok ? 'healthy' : 'unhealthy'}`}
            title={`Backend Service: ${health.message}`}
            role="status"
            aria-live="polite"
          >
            <span className="health-dot" aria-hidden="true"></span>
            <span>{health.ok ? 'Backend Online' : 'Backend Offline'}</span>
          </div>
        </div>
      </div>
    </header>
  );
};
