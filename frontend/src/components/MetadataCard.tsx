import React from 'react';
import { Camera, AlertCircle } from 'lucide-react';
import { ImageMetadata } from '../types/api';

interface MetadataCardProps {
  metadata: ImageMetadata;
}

export const MetadataCard: React.FC<MetadataCardProps> = ({ metadata }) => {
  const format = metadata.image_format || metadata.format;
  const resolution =
    metadata.width && metadata.height ? `${metadata.width} × ${metadata.height} px` : null;

  return (
    <div className="forensic-card" aria-label="Forensic Image Metadata and EXIF">
      <div className="card-title-row">
        <h3 className="card-title">
          <Camera size={18} color="var(--accent-cyan)" />
          Forensic Metadata & EXIF
        </h3>
        <span
          style={{
            fontSize: '0.75rem',
            padding: '0.2rem 0.6rem',
            borderRadius: '9999px',
            backgroundColor: metadata.available ? 'rgba(16, 185, 129, 0.15)' : 'rgba(255, 255, 255, 0.06)',
            color: metadata.available ? '#34d399' : '#94a3b8',
            border: `1px solid ${metadata.available ? 'rgba(16, 185, 129, 0.3)' : 'rgba(255, 255, 255, 0.1)'}`,
            fontWeight: 600,
          }}
        >
          {metadata.available ? 'EXIF Discovered' : 'No EXIF Found'}
        </span>
      </div>

      {metadata.available ? (
        <div className="metadata-table">
          <div className="meta-field">
            <span className="meta-field-label">Container Format</span>
            <span className="meta-field-val">{format || 'Unknown'}</span>
          </div>

          <div className="meta-field">
            <span className="meta-field-label">Dimensions</span>
            <span className="meta-field-val">{resolution || 'N/A'}</span>
          </div>

          <div className="meta-field">
            <span className="meta-field-label">Camera Manufacturer</span>
            <span className="meta-field-val">{metadata.camera_make || 'Not Specified'}</span>
          </div>

          <div className="meta-field">
            <span className="meta-field-label">Camera Model</span>
            <span className="meta-field-val">{metadata.camera_model || 'Not Specified'}</span>
          </div>

          <div className="meta-field">
            <span className="meta-field-label">Processing / Editing Software</span>
            <span className="meta-field-val">{metadata.software || 'None Recorded'}</span>
          </div>

          <div className="meta-field">
            <span className="meta-field-label">Capture Timestamp</span>
            <span className="meta-field-val">{metadata.datetime || 'Not Available'}</span>
          </div>
        </div>
      ) : (
        <div className="metadata-absent-banner">
          <AlertCircle size={20} style={{ flexShrink: 0, marginTop: '2px' }} />
          <div>
            <strong>No EXIF Header Present:</strong> This image contains no embedded EXIF tags.
            This is typical of web re-compressions, social media platforms, screenshots, or pure synthetic generations.
          </div>
        </div>
      )}

      <div className="metadata-forensic-rule">
        <strong>Forensic Principle:</strong> Metadata is treated strictly as supplementary investigative context. Absence or modification of EXIF headers does NOT prove or disprove computational generation.
      </div>
    </div>
  );
};
