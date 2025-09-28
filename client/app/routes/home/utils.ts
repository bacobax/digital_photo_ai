import JSZip from "jszip";

import {
  ACCEPTED_IMAGE_TYPES,
  DEFAULT_API_PATH,
  LOCAL_API_ENDPOINT,
  annotatedRegex,
  balancedRegex,
  faceIdRegex,
  MAX_FILE_SIZE,
} from "./constants";
import type {
  FaceResult,
  FormValuesState,
  ImageAsset,
  Messages,
  NormalizedMeasurement,
  MeasurementLabelKey,
} from "./types";

export function resolveApiEndpoint() {
  if (typeof window === "undefined") {
    return DEFAULT_API_PATH;
  }

  const fromEnv = (import.meta.env as Record<string, string | undefined>)?.VITE_API_URL;
  if (fromEnv && fromEnv.length > 0) {
    return fromEnv.endsWith("/process") ? fromEnv : `${fromEnv.replace(/\/+$/, "")}/process`;
  }

  if (window.location.protocol === "file:") {
    return LOCAL_API_ENDPOINT;
  }

  const isLocalHost = /^(localhost|127\.0\.0\.1)$/.test(window.location.hostname);
  if (import.meta.env.DEV && isLocalHost) {
    return LOCAL_API_ENDPOINT;
  }

  return DEFAULT_API_PATH;
}

export function formatFileSize(size: number) {
  if (size >= 1024 * 1024) {
    return `${(size / (1024 * 1024)).toFixed(1)} MB`;
  }
  if (size >= 1024) {
    return `${(size / 1024).toFixed(1)} KB`;
  }
  return `${size} B`;
}

function formatLabelFromKey(rawKey: string, fallback: string) {
  const normalized = rawKey.replace(/[_-]+/g, " ").replace(/\s+/g, " ");
  const words = normalized.split(" ").filter(Boolean);
  if (words.length === 0) {
    return fallback;
  }
  return words
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

function formatMeasurementValue(
  rawValue: unknown,
  messages: Messages,
  measurementKey: MeasurementLabelKey | null,
) {
  if (typeof rawValue === "number" && Number.isFinite(rawValue)) {
    if (measurementKey === "pixelsPerMm") {
      return `${rawValue.toFixed(2)} ${messages.measurementUnitPxPerMm}`;
    }
    if (measurementKey && measurementKey.includes("chin")) {
      return `${rawValue.toFixed(1)} ${messages.measurementUnitMm}`;
    }
    return `${rawValue.toFixed(0)} ${messages.measurementUnitPx}`;
  }

  if (typeof rawValue === "string" && rawValue.trim().length > 0) {
    return rawValue.trim();
  }

  return "";
}

function buildMeasurements(source: unknown, messages: Messages) {
  const result: NormalizedMeasurement[] = [];

  if (!source || typeof source !== "object") {
    return result;
  }

  const entries = Object.entries(source as Record<string, unknown>);

  for (const [rawKey, rawValue] of entries) {
    if (rawKey.toLowerCase() === "face_id") {
      continue;
    }

    const normalizedKey = rawKey.toLowerCase();

    let measurementKey: MeasurementLabelKey | null = null;
    let tone: NormalizedMeasurement["tone"] = "neutral";

    if (normalizedKey.includes("crown") && normalizedKey.includes("chin")) {
      measurementKey = "crownToChin";
      tone = "primary";
    } else if (normalizedKey.includes("forehead") && normalizedKey.includes("chin")) {
      measurementKey = "foreheadToChin";
      tone = "info";
    } else if ((normalizedKey.includes("hair") || normalizedKey.includes("top")) && normalizedKey.includes("chin")) {
      measurementKey = "hairToChin";
      tone = "warning";
    } else if (normalizedKey.includes("px") && (normalizedKey.includes("mm") || normalizedKey.includes("per"))) {
      measurementKey = "pixelsPerMm";
      tone = "success";
    }

    const label = measurementKey
      ? messages.measurementLabels[measurementKey]
      : formatLabelFromKey(rawKey, messages.measurementFallback);
    const value = formatMeasurementValue(rawValue, messages, measurementKey);

    if (!value) {
      continue;
    }

    result.push({ label, value, tone });
  }

  return result;
}

function extractMeasurementSource(payload: unknown) {
  if (!payload || typeof payload !== "object") {
    return payload;
  }

  const record = payload as Record<string, unknown>;

  if (record.measurements && typeof record.measurements === "object") {
    return record.measurements;
  }

  if (record.metrics && typeof record.metrics === "object") {
    return record.metrics;
  }

  return payload;
}

function resolveFaceIdentifier(face: unknown, index: number) {
  if (!face || typeof face !== "object") {
    return `face${String(index + 1).padStart(2, "0")}`.toLowerCase();
  }

  const record = face as Record<string, unknown>;

  const candidate = record["face_id"] ?? record["faceId"] ?? record["id"] ?? record["identifier"];
  if (typeof candidate === "string" && candidate.trim().length > 0) {
    return candidate.trim().toLowerCase();
  }

  const indexValue = record["index"];
  if (typeof indexValue === "number" && Number.isFinite(indexValue)) {
    const numericIndex = Math.max(1, Math.round(indexValue));
    return `face${String(numericIndex).padStart(2, "0")}`.toLowerCase();
  }

  return `face${String(index + 1).padStart(2, "0")}`.toLowerCase();
}

export function normalizeMetadataFaces(metadata: unknown, messages: Messages) {
  const normalized: Record<string, NormalizedMeasurement[]> = {};

  if (!metadata) {
    return normalized;
  }

  const facesQueue: Array<{ id: string; payload: unknown }> = [];

  if (Array.isArray((metadata as Record<string, unknown>).faces)) {
    const faces = (metadata as Record<string, unknown>).faces as unknown[];
    faces.forEach((face, index) => {
      facesQueue.push({ id: resolveFaceIdentifier(face, index), payload: face });
    });
  } else if (Array.isArray(metadata)) {
    (metadata as unknown[]).forEach((face, index) => {
      facesQueue.push({ id: resolveFaceIdentifier(face, index), payload: face });
    });
  } else {
    facesQueue.push({ id: resolveFaceIdentifier(metadata, 0), payload: metadata });
  }

  facesQueue.forEach(({ id, payload }) => {
    const measurementSource = extractMeasurementSource(payload);
    const metrics = buildMeasurements(measurementSource, messages);
    if (metrics.length > 0) {
      normalized[id] = metrics;
    }
  });

  return normalized;
}

export function extractErrorDetail(payload: unknown): string | null {
  if (!payload) {
    return null;
  }

  if (typeof payload === "string") {
    const trimmed = payload.trim();
    return trimmed.length > 0 ? trimmed : null;
  }

  if (Array.isArray(payload)) {
    for (const item of payload) {
      const detail = extractErrorDetail(item);
      if (detail) {
        return detail;
      }
    }
    return null;
  }

  if (typeof payload === "object") {
    const record = payload as Record<string, unknown>;
    if (typeof record.detail === "string") {
      return record.detail.trim() || null;
    }
    if (Array.isArray(record.detail)) {
      const detail = extractErrorDetail(record.detail);
      if (detail) {
        return detail;
      }
    }
    if (typeof record.message === "string") {
      return record.message.trim() || null;
    }
  }

  return null;
}

export async function parseZipArchive(blob: Blob, messages: Messages) {
  const zip = await JSZip.loadAsync(blob);
  const metadataEntry = zip.file(/metadata\.json$/i)?.[0];
  let metadataJson: unknown = null;

  if (metadataEntry) {
    try {
      const metadataText = await metadataEntry.async("text");
      metadataJson = JSON.parse(metadataText);
    } catch (parseError) {
      console.warn("Unable to parse metadata.json", parseError);
    }
  }

  const metadataByFace = normalizeMetadataFaces(metadataJson, messages);
  const assetsMap = new Map<string, { balanced?: ImageAsset; annotated?: ImageAsset }>();
  const assetPromises: Array<Promise<void>> = [];

  zip.forEach((relativePath, entry) => {
    if (entry.dir) {
      return;
    }

    const lowerPath = relativePath.toLowerCase();
    const faceMatch = lowerPath.match(faceIdRegex);
    const faceId = faceMatch ? faceMatch[1].toLowerCase() : "face01";

    if (balancedRegex.test(lowerPath) || annotatedRegex.test(lowerPath)) {
      const isBalanced = balancedRegex.test(lowerPath);
      const isAnnotated = annotatedRegex.test(lowerPath);

      assetPromises.push(
        entry.async("blob").then((assetBlob) => {
          const filename = relativePath.split("/").pop() ?? `${faceId}.jpg`;
          const asset: ImageAsset = {
            blob: assetBlob,
            url: URL.createObjectURL(assetBlob),
            filename,
          };
          const existing = assetsMap.get(faceId) ?? {};
          if (isBalanced) {
            existing.balanced = asset;
          }
          if (isAnnotated) {
            existing.annotated = asset;
          }
          assetsMap.set(faceId, existing);
        }),
      );
    }
  });

  await Promise.all(assetPromises);

  const faceIds = new Set<string>([...assetsMap.keys(), ...Object.keys(metadataByFace)]);
  if (faceIds.size === 0) {
    faceIds.add("face01");
  }

  const orderedFaceIds = Array.from(faceIds).sort();
  const combinedFaces: FaceResult[] = orderedFaceIds.map((id) => {
    const assets = assetsMap.get(id) ?? {};
    return {
      id,
      balanced: assets.balanced ?? null,
      annotated: assets.annotated ?? null,
      metrics: metadataByFace[id] ?? [],
    };
  });

  return combinedFaces;
}

export function validateIncomingFile(file: File | null) {
  if (!file) {
    return { valid: false, reason: "missing" as const };
  }

  const isKnownType = ACCEPTED_IMAGE_TYPES.includes(file.type);
  const extensionMatches = /\.(jpe?g|png|heic|heif|webp)$/i.test(file.name);

  if (!isKnownType && !extensionMatches) {
    return { valid: false, reason: "type" as const };
  }

  if (file.size > MAX_FILE_SIZE) {
    return { valid: false, reason: "size" as const };
  }

  return { valid: true as const };
}

export function appendFormData(formData: FormData, formValues: FormValuesState) {
  formData.append("target_height_mm", String(formValues.targetHeightMm));
  formData.append("min_height_px", String(formValues.minHeightPx));
  formData.append("min_width_px", String(formValues.minWidthPx));
  formData.append("target_w_over_h", String(formValues.targetWOverH));
  formData.append("top_margin_ratio", String(formValues.topMarginRatio));
  formData.append("bottom_upper_ratio", String(formValues.bottomUpperRatio));
  formData.append("max_crown_to_chin_mm", String(formValues.maxCrownToChinMm));
  formData.append("min_crown_to_chin_mm", String(formValues.minCrownToChinMm));
  formData.append("target_crown_to_chin_mm", String(formValues.targetCrownToChinMm));
  formData.append("max_extra_padding_px", String(formValues.maxExtraPaddingPx));
}

export function revokeFaceObjectUrls(faces: FaceResult[]) {
  faces.forEach((face) => {
    if (face.balanced) {
      URL.revokeObjectURL(face.balanced.url);
    }
    if (face.annotated) {
      URL.revokeObjectURL(face.annotated.url);
    }
  });
}

export { ACCEPTED_IMAGE_TYPES, MAX_FILE_SIZE };
