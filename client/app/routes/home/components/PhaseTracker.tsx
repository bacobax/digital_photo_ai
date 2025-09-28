import classNames from "classnames";

import { PHASE_ORDER } from "../constants";
import type { PhaseTrackerProps } from "../types";

export function PhaseTracker({
  messages,
  statusLabel,
  phaseStatusText,
  progressPercent,
  phaseIndex,
}: PhaseTrackerProps) {
  return (
    <section
      aria-labelledby="status-tracker"
      className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm transition dark:border-slate-800 dark:bg-slate-900"
    >
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 id="status-tracker" className="text-lg font-semibold text-slate-900 dark:text-slate-100">
            {messages.phaseTrackerHeading}
          </h2>
          <p className="text-sm text-slate-500 dark:text-slate-400">{statusLabel}</p>
        </div>
        <div className="flex flex-col items-start gap-2 sm:items-end">
          <span className="text-xs font-medium uppercase tracking-wide text-slate-500 dark:text-slate-400">
            {phaseStatusText}
          </span>
          <div
            role="progressbar"
            aria-label={messages.progressLabel}
            aria-valuenow={progressPercent}
            aria-valuemin={0}
            aria-valuemax={100}
            className="flex h-2 w-48 items-center overflow-hidden rounded-full border border-slate-200 bg-slate-100 dark:border-slate-700 dark:bg-slate-800"
          >
            <span
              className="h-full bg-sky-500 transition-all duration-500 ease-out dark:bg-sky-400"
              style={{ width: `${progressPercent}%` }}
            />
          </div>
        </div>
      </div>
      <ol className="mt-6 grid grid-cols-2 gap-3 text-sm sm:grid-cols-3 lg:grid-cols-7">
        {PHASE_ORDER.map((phase, index) => {
          const isComplete = phaseIndex > index;
          const isActive = phaseIndex === index;
          return (
            <li
              key={phase}
              className={classNames(
                "flex items-center gap-3 rounded-lg border p-3",
                isComplete
                  ? "border-sky-200 bg-sky-50 text-sky-700 dark:border-sky-800 dark:bg-sky-950/40 dark:text-sky-300"
                  : isActive
                    ? "border-sky-400 bg-white text-sky-700 shadow-sm dark:border-sky-500 dark:bg-slate-900 dark:text-sky-300"
                    : "border-slate-200 bg-white text-slate-500 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-400",
              )}
            >
              <span
                className={classNames(
                  "flex h-6 w-6 flex-shrink-0 items-center justify-center rounded-full border text-xs font-semibold",
                  isComplete
                    ? "border-sky-400 bg-sky-500 text-white"
                    : isActive
                      ? "border-sky-500 bg-sky-100 text-sky-700 dark:bg-sky-950/40"
                      : "border-slate-300 bg-slate-100 text-slate-500 dark:border-slate-700 dark:bg-slate-800",
                )}
              >
                {index + 1}
              </span>
              <span className="font-medium">{messages.phaseLabels[phase]}</span>
            </li>
          );
        })}
      </ol>
    </section>
  );
}
