import React from 'react';
import { AlertOctagon, X } from 'lucide-react';

interface ErrorAlertProps {
  message: string;
  onDismiss: () => void;
}

export const ErrorAlert: React.FC<ErrorAlertProps> = ({ message, onDismiss }) => {
  return (
    <div className="error-banner" role="alert" aria-live="assertive">
      <div className="error-content">
        <AlertOctagon size={22} style={{ flexShrink: 0 }} />
        <span>{message}</span>
      </div>
      <button
        type="button"
        onClick={onDismiss}
        style={{
          background: 'none',
          border: 'none',
          color: '#fecdd3',
          cursor: 'pointer',
          padding: '4px',
          display: 'flex',
          alignItems: 'center',
        }}
        aria-label="Dismiss error notice"
      >
        <X size={18} />
      </button>
    </div>
  );
};
