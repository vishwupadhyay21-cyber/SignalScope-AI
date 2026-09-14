import React, { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { Dropzone } from './components/Dropzone';
import { ImagePreview } from './components/ImagePreview';
import { LoadingState } from './components/LoadingState';
import { VerdictBanner } from './components/VerdictBanner';
import { ProbabilityCard } from './components/ProbabilityCard';
import { MetadataCard } from './components/MetadataCard';
import { ExtensibleModules } from './components/ExtensibleModules';
import { ErrorAlert } from './components/ErrorAlert';
import { Footer } from './components/Footer';
import { analyzeImage } from './services/api';
import { AnalysisResponse, ApiError } from './types/api';
import { RefreshCw, Fingerprint, Sparkles } from 'lucide-react';
import './styles/App.css';

export const App: React.FC = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Clean up object URLs to prevent browser memory leaks
  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  const handleFileSelect = (file: File) => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    const newUrl = URL.createObjectURL(file);
    setSelectedFile(file);
    setPreviewUrl(newUrl);
    setResult(null);
    setErrorMessage(null);
  };

  const handleRemove = () => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setSelectedFile(null);
    setPreviewUrl(null);
    setResult(null);
    setErrorMessage(null);
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;

    setIsAnalyzing(true);
    setErrorMessage(null);

    try {
      const response = await analyzeImage(selectedFile);
      setResult(response);
    } catch (err: unknown) {
      const apiErr = err as ApiError;
      setErrorMessage(apiErr?.message || 'An unexpected error occurred during forensic analysis.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleResetForNext = () => {
    handleRemove();
  };

  return (
    <div className="app-wrapper">
      <Header />

      <main className="main-content">
        {/* Hero Section */}
        <section className="hero-section">
          <div className="hero-pill">
            <Sparkles size={13} />
            <span>SIH 2026 Problem Statement</span>
          </div>
          <h1 className="hero-title">
            Telling <span className="hero-gradient">Real From Synthetic</span> in the Age of GenAI
          </h1>
          <p className="hero-description">
            SignalScope delivers multi-signal forensics for images. Perform in-memory integrity validation, extract forensic EXIF parameters, and compute calibrated neural probability scores without saving data to disk.
          </p>
        </section>

        {/* Global Error Banner */}
        {errorMessage && (
          <ErrorAlert message={errorMessage} onDismiss={() => setErrorMessage(null)} />
        )}

        {/* Workflow State 1: Dropzone (No image chosen yet) */}
        {!selectedFile && (
          <Dropzone
            onFileSelect={handleFileSelect}
            onError={(msg) => setErrorMessage(msg)}
            disabled={isAnalyzing}
          />
        )}

        {/* Workflow State 2: Image Preview & Control (Image selected, ready or in progress) */}
        {selectedFile && previewUrl && !result && (
          <ImagePreview
            file={selectedFile}
            previewUrl={previewUrl}
            onAnalyze={handleAnalyze}
            onReplace={() => {
              // Triggering replace clears current and lets dropzone reopen
              const input = document.createElement('input');
              input.type = 'file';
              input.accept = '.jpg,.jpeg,.png,.webp,image/jpeg,image/png,image/webp';
              input.onchange = (e) => {
                const f = (e.target as HTMLInputElement).files?.[0];
                if (f) handleFileSelect(f);
              };
              input.click();
            }}
            onRemove={handleRemove}
            isAnalyzing={isAnalyzing}
          />
        )}

        {/* Loading Indicator */}
        {isAnalyzing && <LoadingState />}

        {/* Workflow State 3: Forensic Results Dashboard */}
        {result && (
          <section className="results-section" aria-label="Forensic Analysis Results">
            {/* Header with Analysis ID & Reset Button */}
            <div className="results-header-bar">
              <div className="analysis-id-badge" title="Unique backend tracking identifier">
                <Fingerprint size={16} color="var(--accent-cyan)" />
                <span>
                  Analysis ID: <code className="font-mono">{result.analysis_id}</code>
                </span>
                {result.filename && (
                  <span style={{ color: 'var(--text-muted)' }}>• {result.filename}</span>
                )}
              </div>

              <button
                type="button"
                className="btn-secondary"
                onClick={handleResetForNext}
                style={{ width: 'auto', padding: '0.5rem 1rem' }}
              >
                <RefreshCw size={15} /> Analyze Another Image
              </button>
            </div>

            {/* Calibrated Verdict Banner */}
            <VerdictBanner
              verdict={result.verdict}
              analysisId={result.analysis_id}
              filename={result.filename}
            />

            {/* Primary Analysis Grid: Probabilities & Metadata */}
            <div className="results-grid">
              <ProbabilityCard probabilities={result.probabilities} />
              <MetadataCard metadata={result.metadata} />
            </div>

            {/* Extensible Future ML Modules (Heatmap, Cues, Attribution, C2PA) */}
            <ExtensibleModules
              visualEvidence={result.visual_evidence}
              explanation={result.explanation}
              attribution={result.generator_attribution}
              robustness={result.robustness}
              provenance={result.provenance}
            />
          </section>
        )}
      </main>

      <Footer />
    </div>
  );
};
