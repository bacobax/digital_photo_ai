import type { Route } from "./+types/home";
import {
  type DragEvent,
  type FormEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import classNames from "classnames";
import FileSaver from "file-saver";

import {
  FooterSection,
  GuidelinesSection,
  HeaderBar,
  OptionsSection,
  PhaseTracker,
  ResultsSection,
  ToastBanner,
  UploadSection,
} from "./home/components";
import { BADGE_STYLES, PHASE_ORDER } from "./home/constants";
import { MESSAGES } from "./home/messages";
import type {
  FaceResult,
  FormValuesState,
  Language,
  Status,
  Toast,
} from "./home/types";
import {
  appendFormData,
  extractErrorDetail,
  parseZipArchive,
  resolveApiEndpoint,
  revokeFaceObjectUrls,
  validateIncomingFile,
} from "./home/utils";

export function meta({}: Route.MetaArgs) {
  return [
    { title: "Digital Photo Assistant" },
    {
      name: "description",
      content: "Upload a photo to receive balanced ID-friendly crops, annotations, and downloads.",
    },
  ];
}

export default function Home() {
  const [language, setLanguage] = useState<Language>("en");
  const messages = useMemo(() => MESSAGES[language], [language]);
  const [highContrast, setHighContrast] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [filePreview, setFilePreview] = useState<string | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [faceResults, setFaceResults] = useState<FaceResult[]>([]);
  const [zipBlob, setZipBlob] = useState<Blob | null>(null);
  const [selectedFaceIndex, setSelectedFaceIndex] = useState(0);
  const [phaseIndex, setPhaseIndex] = useState(0);
  const [isDragActive, setIsDragActive] = useState(false);
  const [toast, setToast] = useState<Toast | null>(null);
  const [formValues, setFormValues] = useState<FormValuesState>({
    target_height_mm: 45,
    min_height_px: 540,
    min_width_px: 420,
    save_debug: false,
    target_w_over_h: 7 / 9,
    top_margin_ratio: 0.1,
    bottom_upper_ratio: 0.8,
    max_crown_to_chin_mm: 36,
    min_crown_to_chin_mm: 31,
    target_crown_to_chin_mm: 34,
    max_extra_padding_px: 600,
    resize_scaling: 0,
  });

  const fileInputRef = useRef<HTMLInputElement>(null);
  const cameraInputRef = useRef<HTMLInputElement>(null);
  const phaseTimerRef = useRef<number | null>(null);
  const processingAreaRef = useRef<HTMLDivElement | null>(null);
  const previousStatusRef = useRef<Status>(status);

  useEffect(() => {
    if (typeof document === "undefined") {
      return;
    }
    const root = document.documentElement;
    const body = document.body;
    root.classList.add("dark");
    body.classList.add("dark");
    root.dataset.theme = "dark";
    body.dataset.theme = "dark";
  }, []);

  const clearPhaseTimer = useCallback(() => {
    if (phaseTimerRef.current !== null) {
      window.clearInterval(phaseTimerRef.current);
      phaseTimerRef.current = null;
    }
  }, []);

  useEffect(() => {
    if (!file) {
      setFilePreview(null);
      return;
    }

    const objectUrl = URL.createObjectURL(file);
    setFilePreview(objectUrl);

    return () => {
      URL.revokeObjectURL(objectUrl);
    };
  }, [file]);

  useEffect(() => {
    return () => {
      revokeFaceObjectUrls(faceResults);
    };
  }, [faceResults]);

  useEffect(() => {
    if (faceResults.length === 0 && selectedFaceIndex !== 0) {
      setSelectedFaceIndex(0);
      return;
    }
    if (selectedFaceIndex > faceResults.length - 1) {
      setSelectedFaceIndex(Math.max(faceResults.length - 1, 0));
    }
  }, [faceResults, selectedFaceIndex]);

  useEffect(() => {
    if (status !== "processing") {
      return;
    }

    setPhaseIndex(0);

    const intervalId = window.setInterval(() => {
      setPhaseIndex((current) => {
        if (current >= PHASE_ORDER.length - 2) {
          return current;
        }
        return current + 1;
      });
    }, 1800);

    phaseTimerRef.current = intervalId;

    return () => {
      window.clearInterval(intervalId);
      phaseTimerRef.current = null;
    };
  }, [status]);

  useEffect(() => {
    if (!toast) {
      return;
    }
    const timeoutId = window.setTimeout(() => {
      setToast(null);
    }, 6000);
    return () => {
      window.clearTimeout(timeoutId);
    };
  }, [toast]);

  useEffect(() => {
    if (status === "processing" && previousStatusRef.current !== "processing") {
      window.requestAnimationFrame(() => {
        processingAreaRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
      });
    }
    previousStatusRef.current = status;
  }, [status]);

  const handleRemoveFile = useCallback(() => {
    setFile(null);
    setStatus("idle");
    setErrorMessage(null);
    setFaceResults([]);
    setZipBlob(null);
    setSelectedFaceIndex(0);
    setPhaseIndex(0);
    setToast(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
    if (cameraInputRef.current) {
      cameraInputRef.current.value = "";
    }
  }, []);

  const handleIncomingFile = useCallback(
    (incoming: File | null) => {
      if (!incoming) {
        return;
      }

      setStatus("validating");
      setErrorMessage(null);

      const validation = validateIncomingFile(incoming);
      if (!validation.valid) {
        setStatus("error");
        setFile(null);
        setErrorMessage(messages.errorValidation);
        setToast({ type: "error", message: messages.toastError });
        return;
      }

      setFile(incoming);
      setFaceResults([]);
      setZipBlob(null);
      setSelectedFaceIndex(0);
      setPhaseIndex(0);
      setStatus("ready");
      setToast(null);

      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
      if (cameraInputRef.current) {
        cameraInputRef.current.value = "";
      }
    },
    [messages.errorValidation, messages.toastError],
  );

  const handleDragOver = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    if (status === "processing") {
      event.dataTransfer.dropEffect = "none";
      return;
    }
    event.dataTransfer.dropEffect = "copy";
    setIsDragActive(true);
  };

  const handleDragLeave = () => {
    setIsDragActive(false);
  };

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragActive(false);
    if (status === "processing") {
      return;
    }
    const incoming = event.dataTransfer.files?.[0];
    handleIncomingFile(incoming ?? null);
  };

  const handleOptionChange = useCallback(
    (field: keyof FormValuesState, value: number) => {
      setFormValues((previous) => ({
        ...previous,
        [field]: Number.isFinite(value) ? value : previous[field],
      }));
    },
    [],
  );

  const handleToggleDebug = useCallback(() => {
    setFormValues((previous) => ({
      ...previous,
      save_debug: !previous.save_debug,
    }));
  }, []);

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    if (!file) {
      setErrorMessage(messages.errorFileRequired);
      setStatus("error");
      setToast({ type: "error", message: messages.toastError });
      return;
    }

    setStatus("processing");
    setErrorMessage(null);
    setToast(null);
    setFaceResults([]);
    setZipBlob(null);
    setSelectedFaceIndex(0);
    setPhaseIndex(0);

    const formData = new FormData();
    formData.append("file", file);
    appendFormData(formData, formValues);
    if (formValues.save_debug) {
      formData.append("save_debug", "true");
    }
    
    console.log({formData})
    try {
      const endpoint = resolveApiEndpoint();
      const response = await fetch(endpoint, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const rawErrorText = await response.text().catch(() => "");
        clearPhaseTimer();

        let serverDetail: string | null = null;
        if (rawErrorText) {
          try {
            serverDetail = extractErrorDetail(JSON.parse(rawErrorText));
          } catch {
            serverDetail = extractErrorDetail(rawErrorText);
          }
        }

        let message = serverDetail ?? messages.errorGeneral;
        if (!serverDetail) {
          if (response.status === 422) {
            message = messages.errorNoFace;
          } else if (response.status >= 500) {
            message = messages.errorNetwork;
          }
        }

        if (rawErrorText) {
          console.error("Processing request returned", response.status, rawErrorText);
        }

        setStatus("error");
        setErrorMessage(message);
        setToast({ type: "error", message: messages.toastError });
        return;
      }

      const blob = await response.blob();
      setZipBlob(blob);

      const faces = await parseZipArchive(blob, messages);
      setFaceResults(faces);
      setSelectedFaceIndex(0);
      clearPhaseTimer();
      setPhaseIndex(PHASE_ORDER.length - 1);
      setStatus("success");
      setToast({ type: "success", message: messages.toastSuccess });
    } catch (networkError) {
      console.error("Processing request failed", networkError);
      clearPhaseTimer();
      setStatus("error");
      setErrorMessage(messages.errorNetwork);
      setToast({ type: "error", message: messages.toastError });
    }
  };

  const handleRetry = () => {
    if (file) {
      setStatus("ready");
      setErrorMessage(null);
      setToast(null);
    } else {
      setStatus("idle");
    }
  };

  const handleDownloadZip = () => {
    if (!zipBlob) {
      return;
    }
    const timestamp = new Date().toISOString().replace(/[:T.-]/g, "").slice(0, 14);
    FileSaver.saveAs(zipBlob, `digital-photo-ai-${timestamp}.zip`);
  };

  const handleDownloadAsset = (asset: FaceResult["balanced"] | FaceResult["annotated"]) => {
    if (!asset) {
      return;
    }
    FileSaver.saveAs(asset.blob, asset.filename);
  };

  const handleOpenFileDialog = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  const handleOpenCameraDialog = useCallback(() => {
    if (!("mediaDevices" in navigator)) {
      setToast({ type: "error", message: messages.cameraNotSupported });
    }
    cameraInputRef.current?.click();
  }, [messages.cameraNotSupported]);

  const canSubmit = useMemo(
    () => Boolean(file) && status !== "processing" && status !== "validating",
    [file, status],
  );

  const statusLabel = useMemo(() => messages.statusLabels[status], [messages, status]);

  const progressPercent = useMemo(
    () => Math.round((phaseIndex / Math.max(PHASE_ORDER.length - 1, 1)) * 100),
    [phaseIndex],
  );

  const phaseStatusText = useMemo(() => {
    const currentPhase = PHASE_ORDER[Math.min(phaseIndex, PHASE_ORDER.length - 1)] ?? "done";
    return `${messages.phaseStatusPrefix}: ${messages.phaseLabels[currentPhase]}`;
  }, [phaseIndex, messages.phaseLabels, messages.phaseStatusPrefix]);

  const multipleFaces = faceResults.length > 1;
  const currentFace = faceResults[selectedFaceIndex] ?? faceResults[0] ?? null;

  const handleSelectPreviousFace = useCallback(() => {
    setSelectedFaceIndex((index) => {
      if (faceResults.length <= 1) {
        return 0;
      }
      return index === 0 ? faceResults.length - 1 : index - 1;
    });
  }, [faceResults.length]);

  const handleSelectNextFace = useCallback(() => {
    setSelectedFaceIndex((index) => {
      if (faceResults.length <= 1) {
        return 0;
      }
      return index === faceResults.length - 1 ? 0 : index + 1;
    });
  }, [faceResults.length]);

  const isUploadSectionCollapsed = status === "processing";

  const containerClass = highContrast
    ? "min-h-screen bg-black text-white transition-colors duration-300"
    : "min-h-screen bg-gray-950 text-gray-100 transition-colors duration-300";

  return (
    <div className={containerClass}>
      <main className="mx-auto flex w-full max-w-5xl flex-col gap-8 px-4 pb-16 pt-10 sm:pt-12 lg:pt-16">
        <HeaderBar
          messages={messages}
          language={language}
          onLanguageChange={setLanguage}
          highContrast={highContrast}
          onToggleContrast={() => setHighContrast((previous) => !previous)}
        />

        <ToastBanner toast={toast} onDismiss={() => setToast(null)} closeLabel={messages.toastClose} />

        <form
          className="grid grid-cols-1 gap-6 lg:grid-cols-[2fr,1fr]"
          onSubmit={handleSubmit}
          aria-busy={status === "processing"}
        >
          <div className="space-y-6">
            <div
              className={classNames(
                "overflow-hidden transition-[max-height,opacity] duration-500 ease-out",
                isUploadSectionCollapsed
                  ? "pointer-events-none max-h-0 opacity-0"
                  : "max-h-[1600px] opacity-100",
              )}
              aria-hidden={isUploadSectionCollapsed}
            >
              <UploadSection
                messages={messages}
                status={status}
                isDragActive={isDragActive}
                file={file}
                filePreview={filePreview}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onOpenFileDialog={handleOpenFileDialog}
                onOpenCameraDialog={handleOpenCameraDialog}
                onFileInputChange={handleIncomingFile}
                onRemoveFile={handleRemoveFile}
                fileInputRef={fileInputRef}
                cameraInputRef={cameraInputRef}
              />
            </div>

            <OptionsSection
              messages={messages}
              formValues={formValues}
              onOptionChange={handleOptionChange}
              onToggleDebug={handleToggleDebug}
              status={status}
              canSubmit={canSubmit}
              onRetry={handleRetry}
            />
          </div>

          <aside className="space-y-6">
            <GuidelinesSection messages={messages} />
          </aside>
        </form>

        <div ref={processingAreaRef}>
          <PhaseTracker
            messages={messages}
            statusLabel={statusLabel}
            phaseStatusText={phaseStatusText}
            progressPercent={progressPercent}
            phaseIndex={phaseIndex}
          />
        </div>

        <ResultsSection
          status={status}
          messages={messages}
          errorMessage={errorMessage}
          multipleFaces={multipleFaces}
          currentFace={currentFace}
          faceResults={faceResults}
          selectedFaceIndex={selectedFaceIndex}
          onSelectPreviousFace={handleSelectPreviousFace}
          onSelectNextFace={handleSelectNextFace}
          onDownloadZip={handleDownloadZip}
          onDownloadAsset={handleDownloadAsset}
          zipBlob={zipBlob}
          badgeStyles={BADGE_STYLES}
        />
      </main>

      <FooterSection messages={messages} language={language} />
    </div>
  );
}
