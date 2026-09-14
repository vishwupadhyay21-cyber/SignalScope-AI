import React from 'react';
import { Layers, Lightbulb, Tag, ShieldAlert, FileCode } from 'lucide-react';
import {
  VisualEvidence,
  Explanation,
  GeneratorAttribution,
  RobustnessInfo,
  ProvenanceInfo,
} from '../types/api';

interface ExtensibleModulesProps {
  visualEvidence?: VisualEvidence | null;
  explanation?: Explanation | null;
  attribution?: GeneratorAttribution | null;
  robustness?: RobustnessInfo | null;
  provenance?: ProvenanceInfo | null;
}

export const ExtensibleModules: React.FC<ExtensibleModulesProps> = ({
  visualEvidence,
  explanation,
  attribution,
  robustness,
  provenance,
}) => {
  // If no extensible modules are present in the response, render nothing cleanly
  const hasHeatmap = Boolean(visualEvidence?.heatmap);
  const hasExplanation = Boolean(explanation?.summary || (explanation?.cues && explanation.cues.length > 0));
  const hasAttribution = Boolean(attribution?.family || attribution?.model_name);
  const hasRobustness = Boolean(robustness?.stability_score !== undefined || robustness?.notes);
  const hasProvenance = Boolean(provenance?.c2pa_present !== undefined || provenance?.issuer);

  if (!hasHeatmap && !hasExplanation && !hasAttribution && !hasRobustness && !hasProvenance) {
    return null;
  }

  return (
    <div className="extensible-grid" aria-label="Supplementary ML Forensic Modules">
      {/* Visual Evidence / Heatmap */}
      {hasHeatmap && visualEvidence?.heatmap && (
        <div className="forensic-card">
          <div className="card-title-row">
            <h3 className="card-title">
              <Layers size={18} color="var(--accent-cyan)" />
              Saliency / Heatmap Localization
            </h3>
            {visualEvidence.saliency_type && (
              <span className="meta-field-label">{visualEvidence.saliency_type}</span>
            )}
          </div>
          <div style={{ borderRadius: 'var(--radius-md)', overflow: 'hidden', background: '#000' }}>
            <img
              src={visualEvidence.heatmap}
              alt="Grad-CAM visual heatmap showing anomalous image regions"
              style={{ width: '100%', maxHeight: '360px', objectFit: 'contain', display: 'block' }}
            />
          </div>
        </div>
      )}

      {/* Human-Readable Explanation */}
      {hasExplanation && (
        <div className="forensic-card">
          <div className="card-title-row">
            <h3 className="card-title">
              <Lightbulb size={18} color="#fbbf24" />
              Faithful Explanation & Cues
            </h3>
          </div>
          {explanation?.summary && (
            <p style={{ fontSize: '0.9rem', color: 'var(--text-primary)', lineHeight: 1.6 }}>
              {explanation.summary}
            </p>
          )}
          {explanation?.cues && explanation.cues.length > 0 && (
            <div className="cues-list">
              {explanation.cues.map((item, idx) => {
                const text = typeof item === 'string' ? item : `${item.category ? `[${item.category}] ` : ''}${item.cue}`;
                return (
                  <div key={idx} className="cue-tag">
                    <Tag size={14} color="var(--accent-cyan)" style={{ flexShrink: 0, marginTop: '2px' }} />
                    <span>{text}</span>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}

      {/* Generator Attribution */}
      {hasAttribution && (
        <div className="forensic-card">
          <div className="card-title-row">
            <h3 className="card-title">
              <Layers size={18} color="#a855f7" />
              Generator Attribution
            </h3>
          </div>
          <div className="metadata-table">
            {attribution?.family && (
              <div className="meta-field">
                <span className="meta-field-label">Generator Family</span>
                <span className="meta-field-val">{attribution.family}</span>
              </div>
            )}
            {attribution?.model_name && (
              <div className="meta-field">
                <span className="meta-field-label">Suspected Model</span>
                <span className="meta-field-val">{attribution.model_name}</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Provenance / C2PA */}
      {hasProvenance && (
        <div className="forensic-card">
          <div className="card-title-row">
            <h3 className="card-title">
              <FileCode size={18} color="#3b82f6" />
              Provenance & Content Credentials
            </h3>
          </div>
          <div className="metadata-table">
            <div className="meta-field">
              <span className="meta-field-label">C2PA Manifest</span>
              <span className="meta-field-val">
                {provenance?.c2pa_present ? 'Valid Manifest Present' : 'No C2PA Manifest Found'}
              </span>
            </div>
            {provenance?.issuer && (
              <div className="meta-field">
                <span className="meta-field-label">Signer / Issuer</span>
                <span className="meta-field-val">{provenance.issuer}</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Robustness */}
      {hasRobustness && (
        <div className="forensic-card">
          <div className="card-title-row">
            <h3 className="card-title">
              <ShieldAlert size={18} color="#ec4899" />
              Robustness Analysis
            </h3>
          </div>
          {robustness?.notes && (
            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>{robustness.notes}</p>
          )}
        </div>
      )}
    </div>
  );
};
