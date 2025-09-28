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
  FaceMarker,
  FaceMarkerKey,
  FaceResult,
  FaceSpan,
  FaceSpanKey,
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
  rawKey: string,
  rawValue: unknown,
  messages: Messages,
  measurementKey: MeasurementLabelKey | null,
) {
  const normalizedKey = rawKey.toLowerCase();
  const indicatesMm =
    measurementKey === "crownToChin" ||
    measurementKey === "foreheadToChin" ||
    measurementKey === "hairToChin" ||
    normalizedKey.includes("mm") ||
    normalizedKey.includes("millimeter") ||
    normalizedKey.includes("millimetre");
  const indicatesPxPerMm = measurementKey === "pixelsPerMm";
  const indicatesPx = normalizedKey.includes("px") || normalizedKey.includes("pixel");
  const indicatesRatio =
    normalizedKey.includes("ratio") ||
    normalizedKey.includes("percent") ||
    normalizedKey.includes("percentage");

  if (typeof rawValue === "number" && Number.isFinite(rawValue)) {
    if (indicatesPxPerMm) {
      return `${rawValue.toFixed(2)} ${messages.measurementUnitPxPerMm}`;
    }
    if (indicatesMm) {
      return `${rawValue.toFixed(1)} ${messages.measurementUnitMm}`;
    }
    if (indicatesPx) {
      return `${rawValue.toFixed(0)} ${messages.measurementUnitPx}`;
    }
    if (indicatesRatio) {
      return rawValue.toFixed(2);
    }
    return rawValue.toString();
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
    const value = formatMeasurementValue(rawKey, rawValue, messages, measurementKey);

    if (!value) {
      continue;
    }

    result.push({ label, value, tone });
  }

  return result;
}

type RawFaceMetadata = Record<string, unknown>;

type NormalizedFaceMetadata = {
  metrics: NormalizedMeasurement[];
  raw: RawFaceMetadata | null;
};

function coerceNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string") {
    const parsed = Number.parseFloat(value);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }
  return null;
}

function pickValue(source: RawFaceMetadata | null, keys: string[]) {
  if (!source) {
    return undefined;
  }
  for (const key of keys) {
    if (key in source) {
      return source[key];
    }
  }
  return undefined;
}

function buildMarkers(raw: RawFaceMetadata | null, messages: Messages, pxPerMm: number | null) {
  if (!raw) {
    return [] as FaceMarker[];
  }

  const markerDefinitions: Array<{ key: FaceMarkerKey; pxKeys: string[]; mmKeys: string[] }> = [
    { key: "crown", pxKeys: ["crown_px", "crownPx"], mmKeys: ["crown_mm", "crownMm"] },
    { key: "forehead", pxKeys: ["forehead_px", "foreheadPx"], mmKeys: ["forehead_mm", "foreheadMm"] },
    { key: "chin", pxKeys: ["chin_px", "chinPx"], mmKeys: ["chin_mm", "chinMm"] },
  ];

  return markerDefinitions
    .map<FaceMarker | null>((definition) => {
      const pxValue = coerceNumber(pickValue(raw, definition.pxKeys));
      const mmFromMetadata = coerceNumber(pickValue(raw, definition.mmKeys));
      if (pxValue === null) {
        return null;
      }
      const millimeters = mmFromMetadata ?? (pxPerMm ? pxValue / pxPerMm : null);
      return {
        key: definition.key,
        label: messages.markerLabels[definition.key],
        distanceFromBottomPx: pxValue,
        distanceFromBottomMm: millimeters,
      } satisfies FaceMarker;
    })
    .filter((marker): marker is FaceMarker => marker !== null);
}

function buildSpans(
  raw: RawFaceMetadata | null,
  markers: FaceMarker[],
  pxPerMm: number | null,
  messages: Messages,
) {
  if (!raw || markers.length === 0) {
    return [] as FaceSpan[];
  }

  const markerMap = new Map<FaceMarkerKey, FaceMarker>(markers.map((marker) => [marker.key, marker]));
  const spanDefinitions: Array<{
    key: FaceSpanKey;
    start: FaceMarkerKey;
    end: FaceMarkerKey;
    mmKeys: string[];
    label: string;
  }> = [
    {
      key: "crownToChin",
      start: "crown",
      end: "chin",
      mmKeys: ["crown_to_chin_mm", "crownToChinMm"],
      label: messages.measurementLabels.crownToChin,
    },
    {
      key: "foreheadToChin",
      start: "forehead",
      end: "chin",
      mmKeys: ["forehead_to_chin_mm", "foreheadToChinMm"],
      label: messages.measurementLabels.foreheadToChin,
    },
  ];

  return spanDefinitions
    .map<FaceSpan | null>((definition) => {
      const startMarker = markerMap.get(definition.start);
      const endMarker = markerMap.get(definition.end);
      if (!startMarker || !endMarker) {
        return null;
      }
      const pxSpan =
        startMarker.distanceFromBottomPx !== null && endMarker.distanceFromBottomPx !== null
          ? Math.abs(startMarker.distanceFromBottomPx - endMarker.distanceFromBottomPx)
          : null;
      const mmFromMetadata = coerceNumber(pickValue(raw, definition.mmKeys));
      const millimeters =
        mmFromMetadata ?? (pxPerMm && pxSpan !== null ? pxSpan / pxPerMm : null);
      if (pxSpan === null && millimeters === null) {
        return null;
      }
      return {
        key: definition.key,
        label: definition.label,
        startKey: definition.start,
        endKey: definition.end,
        pixels: pxSpan,
        millimeters,
      } satisfies FaceSpan;
    })
    .filter((span): span is FaceSpan => span !== null);
}

function buildFaceGuides(raw: RawFaceMetadata | null, messages: Messages) {
  const pxPerMm = coerceNumber(pickValue(raw, ["px_per_mm", "pxPerMm"]));
  const markers = buildMarkers(raw, messages, pxPerMm);
  const spans = buildSpans(raw, markers, pxPerMm, messages);
  return { markers, spans, pxPerMm: pxPerMm ?? null };
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
  const normalized: Record<string, NormalizedFaceMetadata> = {};

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
    const raw = payload && typeof payload === "object" ? (payload as RawFaceMetadata) : null;
    normalized[id] = { metrics, raw };
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
    const metadata = metadataByFace[id];
    const guides = buildFaceGuides(metadata?.raw ?? null, messages);
    return {
      id,
      balanced: assets.balanced ?? null,
      annotated: assets.annotated ?? null,
      metrics: metadata?.metrics ?? [],
      markers: guides.markers,
      spans: guides.spans,
      pxPerMm: guides.pxPerMm,
    } satisfies FaceResult;
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
  formData.append("target_height_mm", String(formValues.target_height_mm));
  formData.append("min_height_px", String(formValues.min_height_px));
  formData.append("min_width_px", String(formValues.min_width_px));
  formData.append("target_w_over_h", String(formValues.target_w_over_h));
  formData.append("top_margin_ratio", String(formValues.top_margin_ratio));
  formData.append("bottom_upper_ratio", String(formValues.bottom_upper_ratio));
  formData.append("max_crown_to_chin_mm", String(formValues.max_crown_to_chin_mm));
  formData.append("min_crown_to_chin_mm", String(formValues.min_crown_to_chin_mm));
  formData.append("target_crown_to_chin_mm", String(formValues.target_crown_to_chin_mm));
  formData.append("max_extra_padding_px", String(formValues.max_extra_padding_px));
  formData.append("resize_scaling", String(formValues.resize_scaling));
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
