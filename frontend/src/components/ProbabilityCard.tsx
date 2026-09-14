import React from 'react';
import { BarChart3, Shield, Cpu } from 'lucide-react';
import { Probabilities } from '../types/api';

interface ProbabilityCardProps {
  probabilities: Probabilities;
}

export const ProbabilityCard: React.FC<ProbabilityCardProps> = ({ probabilities }) => {
  const realPct = Math.round(probabilities.real * 100);
  const aiPct = Math.round(probabilities.ai_generated * 100);

  return (
    <div className="forensic-card" aria-label="Class Probability Distribution">
      <div className="card-title-row">
        <h3 className="card-title">
          <BarChart3 size={18} color="var(--accent-cyan)" />
          Classification Probabilities
        </h3>
        <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Normalized (Σ = 1.0)</span>
      </div>

      <div className="prob-meter-group">
        {/* Synthetic Probability */}
        <div className="prob-item">
          <div className="prob-header">
            <span className="prob-name">
              <Cpu size={15} color="var(--verdict-ai)" />
              Synthetic / AI-Generated
            </span>
            <span className="prob-val" style={{ color: 'var(--verdict-ai)' }}>
              {aiPct}% <small style={{ fontSize: '0.75rem', opacity: 0.8 }}>({probabilities.ai_generated.toFixed(3)})</small>
            </span>
          </div>
          <div className="prob-track" role="progressbar" aria-valuenow={aiPct} aria-valuemin={0} aria-valuemax={100}>
            <div className="prob-fill ai" style={{ width: `${aiPct}%` }}></div>
          </div>
        </div>

        {/* Real Probability */}
        <div className="prob-item">
          <div className="prob-header">
            <span className="prob-name">
              <Shield size={15} color="var(--verdict-real)" />
              Authentic / Real Photo
            </span>
            <span className="prob-val" style={{ color: 'var(--verdict-real)' }}>
              {realPct}% <small style={{ fontSize: '0.75rem', opacity: 0.8 }}>({probabilities.real.toFixed(3)})</small>
            </span>
          </div>
          <div className="prob-track" role="progressbar" aria-valuenow={realPct} aria-valuemin={0} aria-valuemax={100}>
            <div className="prob-fill real" style={{ width: `${realPct}%` }}></div>
          </div>
        </div>
      </div>

      <div className="prob-explanation-note">
        <strong>Forensic Note:</strong> Probabilities reflect the model's Softmax output across visual domain features. A high synthetic score does not constitute legal proof but flags high likelihood of computational synthesis.
      </div>
    </div>
  );
};
