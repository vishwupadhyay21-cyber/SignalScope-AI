/**
 * Centralized API Service for SignalScope Media Forensics
 * Integrates with FastAPI backend (/api/v1/health and /api/v1/analyze)
 */

import { AnalysisResponse, ApiError } from '../types/api';

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(/\/+$/, '');

/**
 * Normalizes backend error messages into actionable, user-friendly notices.
 */
function normalizeErrorMessage(status: number, message?: string): string {
  if (status === 413) {
    return 'The image exceeds the maximum allowed size limit of 10MB. Please upload a smaller file.';
  }
  if (status === 415) {
    return message || 'Unsupported file format. Please provide a standard JPEG, PNG, or WEBP image.';
  }
  if (status === 400) {
    if (message?.includes('corrupted') || message?.includes('not a valid image')) {
      return 'The file appears to be corrupted, truncated, or is not a valid image file.';
    }
    if (message?.includes('dimensions')) {
      return message;
    }
    return message || 'Invalid image file provided. Please verify the file and try again.';
  }
  if (status === 422) {
    return 'The request could not be processed due to validation errors. Ensure an image file was attached.';
  }
  if (status >= 500) {
    return 'An unexpected server error occurred during forensic analysis. Please try again or contact support.';
  }
  return message || `Request failed with status code ${status}.`;
}

/**
 * Checks backend connectivity and health status.
 */
export async function checkBackendHealth(): Promise<{ ok: boolean; message: string }> {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 4000);

    const response = await fetch(`${API_BASE_URL}/api/v1/health`, {
      method: 'GET',
      signal: controller.signal,
      headers: {
        Accept: 'application/json',
      },
    });

    clearTimeout(timeoutId);

    if (response.ok) {
      const data = await response.json();
      if (data?.status === 'healthy') {
        return { ok: true, message: 'Forensics Engine Online' };
      }
    }
    return { ok: false, message: `Engine returned status ${response.status}` };
  } catch (err: unknown) {
    if (err instanceof Error && err.name === 'AbortError') {
      return { ok: false, message: 'Connection timed out' };
    }
    return { ok: false, message: 'Backend unreachable (localhost:8000)' };
  }
}

/**
 * Sends an image file to the backend for AI vs Real classification and metadata extraction.
 *
 * NOTE: Does NOT set Content-Type header manually, letting the browser generate
 * the required multipart/form-data boundary delimiter.
 */
export async function analyzeImage(file: File): Promise<AnalysisResponse> {
  // Client-side quick checks
  if (!file) {
    throw {
      status: 'error',
      code: 400,
      message: 'Please select or drop an image file to analyze.',
    } as ApiError;
  }

  const formData = new FormData();
  // Field name MUST be 'file' as specified in FastAPI route
  formData.append('file', file, file.name);

  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}/api/v1/analyze`, {
      method: 'POST',
      body: formData,
      headers: {
        Accept: 'application/json',
        // Do NOT specify Content-Type: browser sets boundary automatically
      },
    });
  } catch (networkError: unknown) {
    throw {
      status: 'error',
      code: 0,
      message:
        'Cannot connect to the SignalScope forensics backend. Ensure the backend server is running at ' +
        API_BASE_URL,
      raw: networkError,
    } as ApiError;
  }

  let responseBody: any;
  const rawText = await response.text();
  try {
    responseBody = rawText ? JSON.parse(rawText) : {};
  } catch {
    responseBody = { message: rawText };
  }

  if (!response.ok) {
    const backendMessage = responseBody?.message || responseBody?.detail;
    const userMessage = normalizeErrorMessage(response.status, backendMessage);

    const apiError: ApiError = {
      status: 'error',
      code: response.status,
      message: userMessage,
      raw: responseBody,
    };
    throw apiError;
  }

  return responseBody as AnalysisResponse;
}
