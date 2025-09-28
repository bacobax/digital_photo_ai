import classNames from "classnames";
import type { DragEvent } from "react";

import { formatFileSize } from "../utils";
import type { UploadSectionProps } from "../types";

export function UploadSection({
  messages,
  status,
  isDragActive,
  file,
  filePreview,
  onDragOver,
  onDragLeave,
  onDrop,
  onOpenFileDialog,
  onOpenCameraDialog,
  onFileInputChange,
  onRemoveFile,
  fileInputRef,
  cameraInputRef,
}: UploadSectionProps) {
  const handleDragOver = (event: DragEvent<HTMLDivElement>) => {
    onDragOver(event);
  };

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    onDrop(event);
  };

  return (
    <section
      aria-labelledby="upload-section"
      className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm transition dark:border-slate-800 dark:bg-slate-900"
    >
      <div className="flex items-center justify-between gap-4">
        <div>
          <h2 id="upload-section" className="text-lg font-semibold text-slate-900 dark:text-slate-100">
            {messages.uploadHeading}
          </h2>
          <p className="text-sm text-slate-600 dark:text-slate-300">
            {messages.defaultsHint}
          </p>
        </div>
        {file && (
          <button
            type="button"
            onClick={onRemoveFile}
            className="text-sm font-medium text-rose-600 transition hover:text-rose-700 focus:outline-none focus:ring-2 focus:ring-rose-400 dark:text-rose-300 dark:hover:text-rose-200"
          >
            {messages.removeFile}
          </button>
        )}
      </div>

      <div
        role="region"
        aria-label={messages.ariaUploadZone}
        onDragOver={handleDragOver}
        onDragLeave={onDragLeave}
        onDrop={handleDrop}
        className={classNames(
          "mt-4 flex flex-col items-center justify-center rounded-xl border-2 border-dashed px-5 py-10 text-center transition",
          status === "processing"
            ? "cursor-not-allowed border-slate-200 bg-slate-100 text-slate-500 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-400"
            : isDragActive
              ? "border-sky-400 bg-sky-50 text-sky-600 shadow-inner dark:border-sky-500 dark:bg-sky-950/40 dark:text-sky-300"
              : "border-slate-300 bg-slate-50 text-slate-600 dark:border-slate-700 dark:bg-slate-800/70 dark:text-slate-300",
        )}
      >
        <div className="flex flex-col items-center gap-3">
          <span
            className="flex h-14 w-14 items-center justify-center rounded-full border border-slate-200 bg-white text-sky-500 shadow-sm dark:border-slate-700 dark:bg-slate-900"
            aria-hidden="true"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.6"
              className="h-7 w-7"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                d="M4 16v2.5A1.5 1.5 0 0 0 5.5 20h13a1.5 1.5 0 0 0 1.5-1.5V16M16 8l-4-4m0 0L8 8m4-4v12"
              />
            </svg>
          </span>
          <div className="space-y-1">
            <p className="text-base font-medium text-slate-900 dark:text-slate-100">
              {isDragActive ? messages.dragActive : messages.dragDrop}
            </p>
            <p className="text-sm text-slate-500 dark:text-slate-400">
              {messages.dragDropHint}
            </p>
          </div>
          <div className="flex flex-col items-center gap-2 text-sm text-slate-600 dark:text-slate-300">
            <span>{messages.or}</span>
            <div className="flex flex-wrap items-center justify-center gap-2">
              <button
                type="button"
                onClick={onOpenFileDialog}
                disabled={status === "processing"}
                className="rounded-md bg-sky-500 px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-sky-600 focus:outline-none focus:ring-4 focus:ring-sky-200 disabled:cursor-not-allowed disabled:bg-slate-300 dark:bg-sky-600 dark:hover:bg-sky-500 dark:focus:ring-sky-900"
              >
                {messages.chooseFile}
              </button>
              <button
                type="button"
                onClick={onOpenCameraDialog}
                disabled={status === "processing"}
                className="rounded-md border border-slate-300 px-4 py-2 text-sm font-semibold text-slate-700 transition hover:border-sky-400 hover:text-sky-600 focus:outline-none focus:ring-4 focus:ring-sky-200 disabled:cursor-not-allowed disabled:border-slate-200 disabled:text-slate-400 dark:border-slate-700 dark:text-slate-200 dark:hover:border-sky-500 dark:hover:text-sky-300 dark:focus:ring-sky-900"
              >
                {messages.useCamera}
              </button>
            </div>
          </div>
        </div>
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(event) => onFileInputChange(event.target.files?.[0] ?? null)}
        />
        <input
          ref={cameraInputRef}
          type="file"
          accept="image/*"
          capture="environment"
          className="hidden"
          onChange={(event) => onFileInputChange(event.target.files?.[0] ?? null)}
        />
      </div>

      {file && filePreview && (
        <div className="mt-5 flex items-center gap-4 rounded-xl border border-slate-200 bg-slate-50 p-3 dark:border-slate-700 dark:bg-slate-800">
          <img
            src={filePreview}
            alt={messages.previewAlt}
            className="h-16 w-16 rounded-lg object-cover shadow-sm"
          />
          <div className="min-w-0 flex-1 text-left">
            <p className="truncate text-sm font-semibold text-slate-900 dark:text-slate-100">
              {file.name}
            </p>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              {formatFileSize(file.size)}
            </p>
            <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
              {messages.fileNameLabel}
            </p>
          </div>
        </div>
      )}

      <p className="mt-4 text-sm text-slate-500 dark:text-slate-400">
        {messages.privacyNote}
      </p>
    </section>
  );
}
