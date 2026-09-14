/**
 * SignalScope Media Forensics API Types
 * Strictly aligned with FastAPI Backend Schema in app/schemas/analyze.py
 */

export interface Verdict {
  label: string; // "AI-generated" | "Real"
  confidence: number; // 0.0 to 1.0
}

export interface Probabilities {
  real: number; // 0.0 to 1.0
  ai_generated: number; // 0.0 to 1.0
}

export interface ImageMetadata {
  available: boolean;
  image_format?: string | null;
  format?: string | null;
  width?: number | null;
  height?: number | null;
  camera_make?: string | null;
  camera_model?: string | null;
  software?: string | null;
  datetime?: string | null;
}

/** Future ML Bonus Modules (strictly conditionally rendered if present) */
export interface VisualEvidence {
  heatmap?: string | null;
  saliency_type?: string | null;
}

export interface ExplanationCue {
  category?: string;
  cue: string;
  region?: string;
}

export interface Explanation {
  summary?: string | null;
  cues?: (string | ExplanationCue)[] | null;
}

export interface GeneratorAttribution {
  family?: string | null; // e.g. "Diffusion", "GAN", "Auto-regressive"
  model_name?: string | null;
  confidence?: number | null;
}

export interface RobustnessInfo {
  stability_score?: number | null;
  compression_resilience?: string | null;
  notes?: string | null;
}

export interface ProvenanceInfo {
  c2pa_present?: boolean | null;
  issuer?: string | null;
  signature_valid?: boolean | null;
}

export interface AnalysisResponse {
  status: 'success' | string;
  analysis_id: string;
  filename?: string | null;
  verdict: Verdict;
  probabilities: Probabilities;
  metadata: ImageMetadata;

  // Extensible optional fields
  visual_evidence?: VisualEvidence | null;
  explanation?: Explanation | null;
  generator_attribution?: GeneratorAttribution | null;
  robustness?: RobustnessInfo | null;
  provenance?: ProvenanceInfo | null;
}

export interface ApiError {
  status: 'error';
  code: number;
  message: string;
  raw?: unknown;
}
