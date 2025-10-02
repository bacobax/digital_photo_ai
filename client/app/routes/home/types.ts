import type React from "react";

export type Status = "idle" | "validating" | "ready" | "processing" | "success" | "error";

export type PhaseKey =
  | "queued"
  | "processing"
  | "balancing"
  | "detecting"
  | "cropping"
  | "annotating"
  | "done";

export type BadgeTone = "primary" | "info" | "warning" | "success" | "neutral";

export type MeasurementLabelKey =
  | "crownToChin"
  | "foreheadToChin"
  | "hairToChin"
  | "pixelsPerMm";

export type Language = "en" | "it" | "es";

export type PipelineMode = "closed_form" | "legacy";

export type NormalizedMeasurement = {
  label: string;
  value: string;
  tone: BadgeTone;
};

export type ImageAsset = {
  blob: Blob;
  url: string;
  filename: string;
};

export type FaceMarkerKey = "crown" | "forehead" | "chin";

export type FaceMarker = {
  key: FaceMarkerKey;
  label: string;
  distanceFromBottomPx: number | null;
  distanceFromBottomMm: number | null;
};

export type FaceSpanKey = "crownToChin" | "foreheadToChin";

export type FaceSpan = {
  key: FaceSpanKey;
  label: string;
  startKey: FaceMarkerKey;
  endKey: FaceMarkerKey;
  pixels: number | null;
  millimeters: number | null;
};

export type FaceResult = {
  id: string;
  balanced: ImageAsset | null;
  annotated: ImageAsset | null;
  metrics: NormalizedMeasurement[];
  markers: FaceMarker[];
  spans: FaceSpan[];
  pxPerMm: number | null;
  topMarginWarning: TopMarginWarning | null;
};

export type Toast = {
  type: "success" | "error";
  message: string;
};

export type Messages = {
  appTitle: string;
  appSubtitle: string;
  languageLabel: string;
  uploadHeading: string;
  dragDrop: string;
  dragDropHint: string;
  dragActive: string;
  or: string;
  chooseFile: string;
  useCamera: string;
  cameraNotSupported: string;
  privacyNote: string;
  defaultsHint: string;
  optionsHeading: string;
  pipelineHeading: string;
  pipelineClosedFormLabel: string;
  pipelineClosedFormDescription: string;
  pipelineLegacyLabel: string;
  pipelineLegacyDescription: string;
  targetHeightLabel: string;
  targetHeightSuffix: string;
  minHeightLabel: string;
  minWidthLabel: string;
  pixelsSuffix: string;
  saveDebugLabel: string;
  saveDebugDescription: string;
  guidelinesHeading: string;
  validationGuidelines: string;
  guidelines: string[];
  submitButton: string;
  submitDisabled: string;
  retryButton: string;
  uploadAnother: string;
  uploadingSubtitle: string;
  statusLabels: Record<Status, string>;
  phaseTrackerHeading: string;
  phaseLabels: Record<PhaseKey, string>;
  progressLabel: string;
  ariaProcessingStatus: string;
  ariaResultsRegion: string;
  ariaUploadZone: string;
  resultsHeading: string;
  resultsDescriptionSingle: string;
  resultsDescriptionMulti: string;
  metricsHeading: string;
  metadataMissing: string;
  downloadZip: string;
  downloadFinal: string;
  downloadAnnotated: string;
  tipsHeading: string;
  tipsLink: string;
  errorNoFace: string;
  errorNetwork: string;
  errorValidation: string;
  errorGeneral: string;
  errorFileRequired: string;
  toastSuccess: string;
  toastError: string;
  toastClose: string;
  fileNameLabel: string;
  removeFile: string;
  measurementLabels: Record<MeasurementLabelKey, string>;
  measurementUnitMm: string;
  measurementUnitPxPerMm: string;
  measurementUnitPx: string;
  measurementFallback: string;
  markerLabels: Record<FaceMarkerKey, string>;
  faceLabel: string;
  facePosition: string;
  faceSelectorPrev: string;
  faceSelectorNext: string;
  emptyStateTitle: string;
  emptyStateDescription: string;
  downloadDisabled: string;
  loadingPlaceholder: string;
  settingsHeading: string;
  helpHeading: string;
  footerPrivacy: string;
  footerLegal: string;
  footerLanguage: string;
  footerCopyright: string;
  annotatedAlt: string;
  finalAlt: string;
  previewAlt: string;
  languageSwitchAria: string;
  contrastToggleLabel: string;
  contrastToggleOn: string;
  contrastToggleOff: string;
  processingTimeHint: string;
  phaseStatusPrefix: string;
  privacyLinkAria: string;
  helpLinkHint: string;
  ariaAdvancedToggle: string;
  advancedHeading: string;
  advancedSummary: string;
  targetRatioLabel: string;
  topMarginLabel: string;
  bottomUpperLabel: string;
  maxCrownToChinLabel: string;
  minCrownToChinLabel: string;
  targetCrownToChinLabel: string;
  maxExtraPaddingLabel: string;
  minTopMmLabel: string;
  minBottomMmLabel: string;
  shoulderClearanceLabel: string;
  ratioSuffix: string;
  overlayHeading: string;
  overlayLegendHeading: string;
  resizeScalingLabel: string;
  legacyPreviewHeading: string;
  legacyPreviewDescription: string;
  legacyPreviewAspectLabel: string;
  legacyPreviewTopLabel: string;
  legacyPreviewBottomLabel: string;
  closedFormPreviewHeading: string;
  closedFormPreviewDescription: string;
  closedFormPreviewTopLabel: string;
  closedFormPreviewFaceLabel: string;
  closedFormPreviewBottomLabel: string;
  closedFormPreviewTotalLabel: string;
  closedFormPreviewShoulderHint: string;
  topMarginWarningHeading: string;
  topMarginWarningDescription: string;
  topMarginWarningAchieved: string;
  topMarginWarningShoulder: string;
  topMarginWarningSource: string;
};

export type FormValuesState = {
  pipeline: PipelineMode;
  save_debug: boolean;
  target_w_over_h: number;
  top_margin_ratio: number;
  bottom_upper_ratio: number;
  target_height_mm: number;
  min_height_px: number;
  min_width_px: number;
  max_crown_to_chin_mm: number;
  min_crown_to_chin_mm: number;
  target_crown_to_chin_mm: number;
  max_extra_padding_px: number;
  resize_scaling: number;
  min_top_mm: number;
  shoulder_clearance_mm: number;
};

export type NumericFormField = Exclude<keyof FormValuesState, "pipeline" | "save_debug">;

export type BadgeStyles = Record<BadgeTone, string>;

export type PhaseOrder = PhaseKey[];

export type UploadSectionProps = {
  messages: Messages;
  status: Status;
  isDragActive: boolean;
  file: File | null;
  filePreview: string | null;
  onDragOver: (event: React.DragEvent<HTMLDivElement>) => void;
  onDragLeave: () => void;
  onDrop: (event: React.DragEvent<HTMLDivElement>) => void;
  onOpenFileDialog: () => void;
  onOpenCameraDialog: () => void;
  onFileInputChange: (file: File | null) => void;
  onRemoveFile: () => void;
  fileInputRef: React.MutableRefObject<HTMLInputElement | null> | React.RefObject<HTMLInputElement | null>;
  cameraInputRef: React.MutableRefObject<HTMLInputElement | null> | React.RefObject<HTMLInputElement | null>;
};

export type OptionsSectionProps = {
  messages: Messages;
  formValues: FormValuesState;
  onOptionChange: (field: NumericFormField, value: number) => void;
  onPipelineChange: (mode: PipelineMode) => void;
  onToggleDebug: () => void;
  status: Status;
  canSubmit: boolean;
  onRetry: () => void;
};

export type PhaseTrackerProps = {
  messages: Messages;
  statusLabel: string;
  phaseStatusText: string;
  progressPercent: number;
  phaseIndex: number;
};

export type ResultsSectionProps = {
  status: Status;
  messages: Messages;
  errorMessage: string | null;
  multipleFaces: boolean;
  currentFace: FaceResult | null;
  faceResults: FaceResult[];
  selectedFaceIndex: number;
  onSelectPreviousFace: () => void;
  onSelectNextFace: () => void;
  onDownloadZip: () => void;
  onDownloadAsset: (asset: ImageAsset | null) => void;
  zipBlob: Blob | null;
  badgeStyles: BadgeStyles;
};

export type TopMarginWarning = {
  requested: number | null;
  achieved: number | null;
  maxSupported: number | null;
  sourceSupported: number | null;
  exceedsShoulders: boolean;
  exceedsSource: boolean;
};

export type AnnotatedPreviewProps = {
  image: ImageAsset | null;
  markers: FaceMarker[];
  spans: FaceSpan[];
  messages: Messages;
};

export type ToastBannerProps = {
  toast: Toast | null;
  onDismiss: () => void;
  closeLabel: string;
};

export type HeaderBarProps = {
  messages: Messages;
  language: Language;
  onLanguageChange: (language: Language) => void;
  highContrast: boolean;
  onToggleContrast: () => void;
};

export type GuidelinesSectionProps = {
  messages: Messages;
};

export type FooterSectionProps = {
  messages: Messages;
  language: Language;
};
