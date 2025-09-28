import classNames from "classnames";

import { SUPPORTED_LANGUAGES } from "../constants";
import type { HeaderBarProps, Language } from "../types";

export function HeaderBar({
  messages,
  language,
  onLanguageChange,
  highContrast,
  onToggleContrast,
}: HeaderBarProps) {
  return (
    <header className="flex flex-col gap-6 lg:flex-row lg:items-start lg:justify-between">
      <div className="space-y-2">
        <h1 className="text-3xl font-semibold tracking-tight sm:text-4xl">
          {messages.appTitle}
        </h1>
        <p className="max-w-2xl text-base text-slate-600 dark:text-slate-300">
          {messages.appSubtitle}
        </p>
      </div>
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
        <label className="flex items-center gap-2 text-sm font-medium text-slate-700 dark:text-slate-200">
          <span className="sr-only">{messages.languageLabel}</span>
          <select
            aria-label={messages.languageSwitchAria}
            className="rounded-md border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 shadow-sm focus:border-sky-500 focus:outline-none focus:ring-4 focus:ring-sky-200 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100 dark:focus:border-sky-400 dark:focus:ring-sky-950"
            value={language}
            onChange={(event) => onLanguageChange(event.target.value as Language)}
          >
            {SUPPORTED_LANGUAGES.map(({ code, label, nativeName }) => (
              <option key={code} value={code}>
                {label} · {nativeName}
              </option>
            ))}
          </select>
        </label>
        <button
          type="button"
          role="switch"
          aria-checked={highContrast}
          onClick={onToggleContrast}
          className={classNames(
            "inline-flex items-center gap-2 rounded-md px-3 py-2 text-sm font-medium transition-colors focus:outline-none focus:ring-4 focus:ring-sky-200 dark:focus:ring-sky-900",
            highContrast
              ? "bg-slate-900 text-white dark:bg-white dark:text-slate-900"
              : "bg-slate-200 text-slate-900 dark:bg-slate-800 dark:text-slate-100",
          )}
        >
          <span>{messages.contrastToggleLabel}</span>
          <span className="rounded bg-white/70 px-2 py-0.5 text-xs font-semibold text-slate-900 dark:bg-slate-900/60 dark:text-slate-100">
            {highContrast ? messages.contrastToggleOn : messages.contrastToggleOff}
          </span>
        </button>
      </div>
    </header>
  );
}
