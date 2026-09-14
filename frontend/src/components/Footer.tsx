import React from 'react';
import { Info, Lock, Award } from 'lucide-react';

export const Footer: React.FC = () => {
  return (
    <footer className="app-footer" role="contentinfo">
      <div className="footer-container">
        <div className="disclaimer-card">
          <Info size={22} className="disclaimer-icon" aria-hidden="true" />
          <div className="disclaimer-text">
            <strong>SIH 2026 Ethical & Forensic Protocol:</strong> SignalScope performs probabilistic signal and artifact analysis for media integrity verification. Predictions represent calibrated likelihood assessments (&ldquo;Likely AI-generated&rdquo;) based on neural feature distributions and do not constitute absolute proof or accusation. This system adheres strictly to the SIH Problem Statement 2 scope: general synthetic media evaluation with zero profiling or facial biometric targeting.
          </div>
        </div>

        <div className="footer-meta-row">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Lock size={14} color="var(--accent-cyan)" />
            <span>Zero-Retention Privacy: Images are processed exclusively in-memory without persistent disk caching or external tracking.</span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Award size={14} color="var(--verdict-real)" />
            <span>Smart India Hackathon 2026 • PS-2 Media Forensics</span>
          </div>
        </div>
      </div>
    </footer>
  );
};
