import classNames from "classnames";

import type { ResultsSectionProps } from "../types";

export function ResultsSection({
  status,
  messages,
  errorMessage,
  multipleFaces,
  currentFace,
  faceResults,
  selectedFaceIndex,
  onSelectPreviousFace,
  onSelectNextFace,
  onDownloadZip,
  onDownloadAsset,
  zipBlob,
  badgeStyles,
}: ResultsSectionProps) {
  return (
    <section
      aria-labelledby="results-section"
      aria-live="polite"
      aria-busy={status === "processing"}
      className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm transition dark:border-slate-800 dark:bg-slate-900"
    >
      <div className="flex flex-col gap-2">
        <h2 id="results-section" className="text-lg font-semibold text-slate-900 dark:text-slate-100">
          {messages.resultsHeading}
        </h2>
        <p className="text-sm text-slate-600 dark:text-slate-300">
          {multipleFaces ? messages.resultsDescriptionMulti : messages.resultsDescriptionSingle}
        </p>
      </div>

      {errorMessage && (
        <div
          role="alert"
          className="mt-4 rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700 dark:border-rose-900 dark:bg-rose-950 dark:text-rose-200"
        >
          {errorMessage}
        </div>
      )}

      {status === "idle" && (
        <div className="mt-6 rounded-xl border border-dashed border-slate-300 bg-slate-50 p-6 text-center text-sm text-slate-500 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-300">
          <p className="text-base font-semibold text-slate-700 dark:text-slate-100">
            {messages.emptyStateTitle}
          </p>
          <p className="mt-1 text-sm">{messages.emptyStateDescription}</p>
        </div>
      )}

      {(status === "ready" || status === "validating") && !errorMessage && (
        <div className="mt-6 rounded-xl border border-dashed border-sky-200 bg-sky-50 p-6 text-sm text-sky-700 dark:border-sky-800 dark:bg-sky-950/40 dark:text-sky-300">
          {messages.validationGuidelines}
        </div>
      )}

      {status === "processing" && (
        <div className="mt-6 grid gap-4 sm:grid-cols-2">
          {[0, 1].map((index) => (
            <div
              key={index}
              className="h-52 animate-pulse rounded-xl border border-slate-200 bg-slate-100/80 dark:border-slate-700 dark:bg-slate-800"
            />
          ))}
          <div className="sm:col-span-2">
            <div className="h-16 animate-pulse rounded-xl border border-slate-200 bg-slate-100/80 dark:border-slate-700 dark:bg-slate-800" />
          </div>
        </div>
      )}

      {status === "success" && currentFace && (
        <div className="mt-6 space-y-6">
          {multipleFaces && (
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="text-sm font-medium text-slate-600 dark:text-slate-300">
                {messages.faceLabel}: {messages.facePosition
                  .replace("{current}", String(selectedFaceIndex + 1))
                  .replace("{total}", String(faceResults.length))}
              </div>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={onSelectPreviousFace}
                  aria-label={messages.faceSelectorPrev}
                  className="rounded-full border border-slate-200 p-2 text-slate-600 transition hover:border-sky-400 hover:text-sky-600 focus:outline-none focus:ring-2 focus:ring-sky-400 dark:border-slate-700 dark:text-slate-300 dark:hover:border-sky-500 dark:hover:text-sky-300"
                >
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    viewBox="0 0 20 20"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.6"
                    className="h-4 w-4"
                  >
                    <path d="M12 5 7 10l5 5" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                </button>
                <button
                  type="button"
                  onClick={onSelectNextFace}
                  aria-label={messages.faceSelectorNext}
                  className="rounded-full border border-slate-200 p-2 text-slate-600 transition hover:border-sky-400 hover:text-sky-600 focus:outline-none focus:ring-2 focus:ring-sky-400 dark:border-slate-700 dark:text-slate-300 dark:hover:border-sky-500 dark:hover:text-sky-300"
                >
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    viewBox="0 0 20 20"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.6"
                    className="h-4 w-4"
                  >
                    <path d="m8 5 5 5-5 5" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                </button>
              </div>
            </div>
          )}

          <div className="grid gap-6 md:grid-cols-2">
            <figure className="rounded-2xl border border-slate-200 bg-slate-50 p-4 text-center shadow-inner dark:border-slate-700 dark:bg-slate-800">
              <figcaption className="mb-3 text-sm font-semibold text-slate-600 dark:text-slate-300">
                {messages.downloadFinal}
              </figcaption>
              {currentFace.balanced ? (
                <img
                  src={currentFace.balanced.url}
                  alt={messages.finalAlt}
                  className="mx-auto aspect-[35/45] w-full max-w-xs rounded-lg border border-slate-200 object-cover shadow-sm dark:border-slate-700"
                />
              ) : (
                <div className="flex h-48 items-center justify-center rounded-lg border border-dashed border-slate-300 text-sm text-slate-500 dark:border-slate-700 dark:text-slate-400">
                  {messages.loadingPlaceholder}
                </div>
              )}
              <button
                type="button"
                onClick={() => onDownloadAsset(currentFace.balanced)}
                disabled={!currentFace.balanced}
                className="mt-4 inline-flex items-center justify-center gap-2 rounded-md bg-sky-600 px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-sky-700 focus:outline-none focus:ring-4 focus:ring-sky-200 disabled:cursor-not-allowed disabled:bg-slate-300 dark:bg-sky-500 dark:hover:bg-sky-400 dark:focus:ring-sky-900"
              >
                {messages.downloadFinal}
              </button>
            </figure>
            <figure className="rounded-2xl border border-slate-200 bg-slate-50 p-4 text-center shadow-inner dark:border-slate-700 dark:bg-slate-800">
              <figcaption className="mb-3 text-sm font-semibold text-slate-600 dark:text-slate-300">
                {messages.downloadAnnotated}
              </figcaption>
              {currentFace.annotated ? (
                <img
                  src={currentFace.annotated.url}
                  alt={messages.annotatedAlt}
                  className="mx-auto aspect-[35/45] w-full max-w-xs rounded-lg border border-slate-200 object-cover shadow-sm dark:border-slate-700"
                />
              ) : (
                <div className="flex h-48 items-center justify-center rounded-lg border border-dashed border-slate-300 text-sm text-slate-500 dark:border-slate-700 dark:text-slate-400">
                  {messages.loadingPlaceholder}
                </div>
              )}
              <button
                type="button"
                onClick={() => onDownloadAsset(currentFace.annotated)}
                disabled={!currentFace.annotated}
                className="mt-4 inline-flex items-center justify-center gap-2 rounded-md bg-slate-900 px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-slate-700 focus:outline-none focus:ring-4 focus:ring-slate-400 disabled:cursor-not-allowed disabled:bg-slate-300 dark:bg-slate-700 dark:hover:bg-slate-600 dark:focus:ring-slate-900"
              >
                {messages.downloadAnnotated}
              </button>
            </figure>
          </div>

          <div className="rounded-2xl border border-slate-200 bg-slate-50 p-5 dark:border-slate-700 dark:bg-slate-800">
            <h3 className="text-base font-semibold text-slate-700 dark:text-slate-200">
              {messages.metricsHeading}
            </h3>
            {currentFace.metrics.length > 0 ? (
              <div className="mt-3 flex flex-wrap gap-2">
                {currentFace.metrics.map((metric) => (
                  <span
                    key={`${currentFace.id}-${metric.label}-${metric.value}`}
                    className={classNames(
                      "inline-flex items-center gap-2 rounded-full px-3 py-1 text-xs font-semibold",
                      badgeStyles[metric.tone],
                    )}
                  >
                    <span className="h-2 w-2 rounded-full bg-current" aria-hidden="true" />
                    {metric.label}
                    <span className="font-normal">{metric.value}</span>
                  </span>
                ))}
              </div>
            ) : (
              <p className="mt-3 text-sm text-slate-500 dark:text-slate-300">{messages.metadataMissing}</p>
            )}
          </div>

          <div className="flex flex-wrap items-center justify-between gap-3">
            <button
              type="button"
              onClick={onDownloadZip}
              disabled={!zipBlob}
              className="inline-flex items-center justify-center gap-2 rounded-md bg-emerald-600 px-5 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-emerald-700 focus:outline-none focus:ring-4 focus:ring-emerald-200 disabled:cursor-not-allowed disabled:bg-slate-300 dark:bg-emerald-500 dark:hover:bg-emerald-400 dark:focus:ring-emerald-900"
            >
              {messages.downloadZip}
            </button>
            {!zipBlob && <p className="text-xs text-slate-500 dark:text-slate-400">{messages.downloadDisabled}</p>}
          </div>
        </div>
      )}
    </section>
  );
}
