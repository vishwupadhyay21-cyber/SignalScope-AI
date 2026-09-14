import React, { useRef, useState, useEffect } from 'react';
import { UploadCloud } from 'lucide-react';

interface DropzoneProps {
  onFileSelect: (file: File) => void;
  onError: (msg: string) => void;
  disabled?: boolean;
}

const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/webp'];
const MAX_SIZE_MB = 10;
const MAX_SIZE_BYTES = MAX_SIZE_MB * 1024 * 1024;

// Floating code symbols shown inside/around the dropzone
const CODE_SYMBOLS = [
  '01', 'FF', 'RGB', '<>', '//', '{}', '0x', 'PNG',
  '==', '=>', '&&', '[]', '~~', 'AI', '!?', 'ML',
  'px', '::',  ';', '\\n', '...', 'EOF',
];

export const Dropzone: React.FC<DropzoneProps> = ({ onFileSelect, onError, disabled = false }) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const [statusTick, setStatusTick] = useState(0);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Cycle the status line text
  const STATUS_LINES = [
    'AWAITING_FILE_INPUT...',
    'SYSTEM_READY :: IDLE',
    'FORENSIC_ENGINE :: ONLINE',
    'DROP_TARGET :: ACTIVE',
  ];

  useEffect(() => {
    const id = setInterval(() => setStatusTick(t => (t + 1) % STATUS_LINES.length), 2400);
    return () => clearInterval(id);
  }, []);

  const validateAndForward = (file: File) => {
    if (!ALLOWED_TYPES.includes(file.type.toLowerCase())) {
      const ext = file.name.split('.').pop()?.toUpperCase() || 'UNKNOWN';
      onError(`Unsupported image format '${ext}'. Allowed formats are JPEG, PNG, and WEBP.`);
      return;
    }
    if (file.size > MAX_SIZE_BYTES) {
      onError(`File size (${(file.size / (1024 * 1024)).toFixed(1)}MB) exceeds the maximum limit of ${MAX_SIZE_MB}MB.`);
      return;
    }
    if (file.size === 0) {
      onError('The selected file is empty (0 bytes). Please upload a valid image.');
      return;
    }
    onFileSelect(file);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
    if (disabled) return;
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      validateAndForward(e.dataTransfer.files[0]);
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (!disabled) setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleClick = () => {
    if (!disabled && fileInputRef.current) {
      fileInputRef.current.value = '';
      fileInputRef.current.click();
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if ((e.key === 'Enter' || e.key === ' ') && !disabled) {
      e.preventDefault();
      handleClick();
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      validateAndForward(e.target.files[0]);
    }
  };

  return (
    <div
      className={`dropzone-container ${isDragOver ? 'is-dragover' : ''}`}
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onClick={handleClick}
      onKeyDown={handleKeyDown}
      role="button"
      tabIndex={disabled ? -1 : 0}
      aria-label="Upload image for AI forensics analysis. Drag and drop or press enter to browse."
      aria-disabled={disabled}
    >
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleInputChange}
        accept=".jpg,.jpeg,.png,.webp,image/jpeg,image/png,image/webp"
        style={{ display: 'none' }}
        disabled={disabled}
        aria-hidden="true"
      />

      {/* === FLOATING CODE SYMBOLS (background layer) === */}
      <div className="dz-float-symbols" aria-hidden="true">
        {CODE_SYMBOLS.map((sym, i) => (
          <span key={i} className="dz-float-sym" style={{ '--i': i } as React.CSSProperties}>
            {sym}
          </span>
        ))}
      </div>

      {/* === 4 CORNER BRACKETS === */}
      <div className="dz-corner dz-corner--tl" aria-hidden="true" />
      <div className="dz-corner dz-corner--tr" aria-hidden="true" />
      <div className="dz-corner dz-corner--bl" aria-hidden="true" />
      <div className="dz-corner dz-corner--br" aria-hidden="true" />

      {/* === HORIZONTAL SCAN BEAM === */}
      <div className="dz-scan-beam" aria-hidden="true" />

      {/* === RADAR / SONAR PULSE RINGS === */}
      <div className="dz-radar" aria-hidden="true">
        <div className="dz-radar-ring dz-radar-ring--1" />
        <div className="dz-radar-ring dz-radar-ring--2" />
        <div className="dz-radar-ring dz-radar-ring--3" />

        {/* Central icon */}
        <div className="dropzone-icon-box">
          <UploadCloud size={32} />
        </div>
      </div>

      {/* === TEXT GROUP === */}
      <div className="dropzone-text-group">
        <div className="dropzone-title">
          Drop suspicious image here, or{' '}
          <span className="dropzone-highlight">browse file</span>
        </div>
        <div className="dropzone-spec">
          Strict forensic in-memory processing • Maximum image size: {MAX_SIZE_MB}MB
        </div>
      </div>

      {/* === FORMAT TAGS === */}
      <div className="dropzone-tags" aria-label="Supported file formats">
        <span className="format-tag">JPEG / JPG</span>
        <span className="format-tag">PNG</span>
        <span className="format-tag">WEBP</span>
      </div>

      {/* === CYCLING STATUS LINE === */}
      <div className="dz-status-line" aria-live="polite" aria-atomic="true">
        <span className="dz-status-prompt">$</span>
        <span className="dz-status-text">{STATUS_LINES[statusTick]}</span>
        <span className="dz-status-cursor" aria-hidden="true">█</span>
      </div>
    </div>
  );
};
