import React from 'react';

export const LoadingState: React.FC = () => {
  return (
    <div className="loading-box" role="status" aria-live="polite">
      <div className="spinner-forensic" aria-hidden="true">
        <div className="spinner-outer"></div>
        <div className="spinner-inner"></div>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
        <h4 style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--text-primary)' }}>
          Forensic Processing in Progress
        </h4>
        <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
          Performing in-memory file integrity verification, forensic EXIF extraction, and detector inference.
        </p>
      </div>

      <div className="loading-steps" aria-hidden="true">
        <div className="loading-step-item">
          <span className="step-indicator"></span>
          <span>Validating pixel integrity & color channels (RGB)</span>
        </div>
        <div className="loading-step-item">
          <span className="step-indicator"></span>
          <span>Extracting forensic camera & software metadata tags</span>
        </div>
        <div className="loading-step-item">
          <span className="step-indicator"></span>
          <span>Computing classifier probabilities & calibrated confidence</span>
        </div>
      </div>
    </div>
  );
};
