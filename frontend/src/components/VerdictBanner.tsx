import React from 'react';
import { AlertTriangle, CheckCircle2 } from 'lucide-react';
import { Verdict } from '../types/api';

interface VerdictBannerProps {
  verdict: Verdict;
  analysisId?: string;
  filename?: string | null;
}

export const VerdictBanner: React.FC<VerdictBannerProps> = ({ verdict }) => {
  const isAi = verdict.label.toLowerCase().includes('ai');
  const confidencePercent = Math.round(verdict.confidence * 100);

  // Calibrated responsible framing as required by SIH PS-2
  const verdictHeading = isAi ? 'Likely AI-Generated' : 'Likely Authentic / Real';

  return (
    <section className={`verdict-banner ${isAi ? 'verdict-ai' : 'verdict-real'}`} aria-label="Analysis Verdict">
      <div className="verdict-main-info">
        <div className="verdict-icon-container" aria-hidden="true">
          {isAi ? <AlertTriangle size={36} /> : <CheckCircle2 size={36} />}
        </div>

        <div className="verdict-text-block">
          <span className="verdict-subheading">Model Assessment Result</span>
          <h2 className="verdict-title">{verdictHeading}</h2>
          <p className="verdict-framing-note">
            {isAi
              ? 'Model-based statistical evaluation indicates structural signal patterns and frequency artifacts typical of synthetic image generation.'
              : 'Statistical pattern analysis found consistency with camera sensor pipelines without predominant synthetic generator artifacts.'}
          </p>
        </div>
      </div>

      <div className="verdict-score-block">
        <span className="confidence-label">Calibrated Confidence</span>
        <div className="confidence-chip" title={`Model Confidence: ${verdict.confidence.toFixed(4)}`}>
          <span>{confidencePercent}%</span>
        </div>
        <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
          Score: {verdict.confidence.toFixed(2)} / 1.00
        </span>
      </div>
    </section>
  );
};
