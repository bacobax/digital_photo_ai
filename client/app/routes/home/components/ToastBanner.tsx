import classNames from "classnames";

import type { ToastBannerProps } from "../types";

export function ToastBanner({ toast, onDismiss, closeLabel }: ToastBannerProps) {
  if (!toast) {
    return null;
  }

  return (
    <div
      role="status"
      aria-live="polite"
      className={classNames(
        "fixed inset-x-0 top-3 z-20 mx-auto w-[calc(100%-2rem)] max-w-xl rounded-lg border px-4 py-3 shadow-lg sm:px-5",
        toast.type === "success"
          ? "border-emerald-200 bg-emerald-50 text-emerald-900 dark:border-emerald-800 dark:bg-emerald-950 dark:text-emerald-100"
          : "border-rose-200 bg-rose-50 text-rose-900 dark:border-rose-800 dark:bg-rose-950 dark:text-rose-100",
      )}
    >
      <div className="flex items-start gap-3">
        <span className="mt-0.5 inline-block h-2 w-2 rounded-full bg-current" aria-hidden="true" />
        <p className="flex-1 text-sm font-medium">{toast.message}</p>
        <button
          type="button"
          onClick={onDismiss}
          className="rounded-md px-2 py-1 text-xs font-semibold text-slate-600 transition-colors hover:bg-white/40 hover:text-slate-900 focus:outline-none focus:ring-2 focus:ring-sky-400 dark:text-slate-200 dark:hover:bg-slate-800/70 dark:hover:text-white"
        >
          {closeLabel}
        </button>
      </div>
    </div>
  );
}
