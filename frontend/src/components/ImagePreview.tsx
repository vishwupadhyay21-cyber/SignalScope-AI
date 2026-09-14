import React, { useEffect, useState } from 'react';
import { Play, RefreshCw, Trash2, FileCheck } from 'lucide-react';

interface ImagePreviewProps {
  file: File;
  previewUrl: string;
  onAnalyze: () => void;
  onReplace: () => void;
  onRemove: () => void;
  isAnalyzing: boolean;
}

export const ImagePreview: React.FC<ImagePreviewProps> = ({
  file,
  previewUrl,
  onAnalyze,
  onReplace,
  onRemove,
  isAnalyzing,
}) => {
  const [dimensions, setDimensions] = useState<{ width: number; height: number } | null>(null);

  useEffect(() => {
    const img = new Image();
    img.onload = () => {
      setDimensions({ width: img.naturalWidth, height: img.naturalHeight });
    };
    img.src = previewUrl;
  }, [previewUrl]);

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  return (
    <div className="preview-card" aria-label="Selected image preview and actions">
      <div className="preview-grid">
        <div className="preview-viewport">
          <img
            src={previewUrl}
            alt="Preview of uploaded media to analyze"
            className="preview-image"
          />
          {isAnalyzing && (
            <div className="scan-overlay" aria-hidden="true">
              <div className="scan-line"></div>
            </div>
          )}
        </div>

        <div className="preview-sidebar">
          <div className="file-meta-header">
            <span className="meta-badge">
              <FileCheck size={14} /> Ready for Forensic Scan
            </span>
            <h3 className="file-name-display" title={file.name}>
              {file.name}
            </h3>

            <div className="file-stat-row">
              <div className="file-stat-item">
                <span className="stat-label">File Size</span>
                <span className="stat-value">{formatFileSize(file.size)}</span>
              </div>
              <div className="file-stat-item">
                <span className="stat-label">MIME Type</span>
                <span className="stat-value">{file.type || 'image/*'}</span>
              </div>
              {dimensions && (
                <div className="file-stat-item">
                  <span className="stat-label">Resolution</span>
                  <span className="stat-value">
                    {dimensions.width} × {dimensions.height}
                  </span>
                </div>
              )}
            </div>
          </div>

          <div className="action-button-group">
            <button
              type="button"
              className="btn-primary"
              onClick={onAnalyze}
              disabled={isAnalyzing}
              aria-busy={isAnalyzing}
            >
              {isAnalyzing ? (
                <>
                  <RefreshCw size={18} className="spinner-outer" /> Analyzing Signal Patterns...
                </>
              ) : (
                <>
                  <Play size={18} fill="currentColor" /> Run Forensic Analysis
                </>
              )}
            </button>

            <button
              type="button"
              className="btn-secondary"
              onClick={onReplace}
              disabled={isAnalyzing}
              aria-label="Replace selected image with another file"
            >
              <RefreshCw size={16} /> Choose Another Image
            </button>

            <button
              type="button"
              className="btn-secondary btn-danger"
              onClick={onRemove}
              disabled={isAnalyzing}
              aria-label="Remove image"
            >
              <Trash2 size={16} /> Discard
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
