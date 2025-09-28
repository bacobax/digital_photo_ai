import type { BadgeStyles, Language, PhaseOrder } from "./types";

export const DEFAULT_API_PATH = "/process";
export const LOCAL_API_ENDPOINT = "http://127.0.0.1:8000/process";
export const MAX_FILE_SIZE = 12 * 1024 * 1024;
export const ACCEPTED_IMAGE_TYPES = [
  "image/jpeg",
  "image/png",
  "image/heic",
  "image/heif",
  "image/webp",
];

export const PHASE_ORDER: PhaseOrder = [
  "queued",
  "processing",
  "balancing",
  "detecting",
  "cropping",
  "annotating",
  "done",
];

export const BADGE_STYLES: BadgeStyles = {
  primary:
    "bg-sky-100 text-sky-900 dark:bg-sky-950 dark:text-sky-100 border border-sky-200 dark:border-sky-800",
  info: "bg-teal-100 text-teal-900 dark:bg-teal-950 dark:text-teal-100 border border-teal-200 dark:border-teal-800",
  warning:
    "bg-amber-100 text-amber-900 dark:bg-amber-950 dark:text-amber-100 border border-amber-200 dark:border-amber-800",
  success:
    "bg-emerald-100 text-emerald-900 dark:bg-emerald-950 dark:text-emerald-100 border border-emerald-200 dark:border-emerald-800",
  neutral:
    "bg-slate-100 text-slate-900 dark:bg-slate-800 dark:text-slate-100 border border-slate-200 dark:border-slate-700",
};

export const SUPPORTED_LANGUAGES: Array<{ code: Language; label: string; nativeName: string }> = [
  { code: "en", label: "EN", nativeName: "English" },
  { code: "it", label: "IT", nativeName: "Italiano" },
  { code: "es", label: "ES", nativeName: "Espanol" },
];

export const balancedRegex = /final_balanced/i;
export const annotatedRegex = /final_annotated/i;
export const faceIdRegex = /(face\d+)/i;
