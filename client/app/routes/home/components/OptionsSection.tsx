import { useState } from "react";
import classNames from "classnames";

import type { OptionsSectionProps } from "../types";

export function OptionsSection({
  messages,
  formValues,
  onOptionChange,
  onToggleDebug,
  status,
  canSubmit,
  onRetry,
}: OptionsSectionProps) {
  const [showAdvanced, setShowAdvanced] = useState(false);

  const disabled = status === "processing";

  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm transition dark:border-slate-800 dark:bg-slate-900">
      <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100">
        {messages.optionsHeading}
      </h3>
      <p className="mt-1 text-sm text-slate-600 dark:text-slate-300">
        {messages.validationGuidelines}
      </p>
      <div className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2">
        <NumberInput
          label={messages.targetHeightLabel}
          suffix={messages.targetHeightSuffix}
          value={formValues.targetHeightMm}
          onChange={(value) => onOptionChange("targetHeightMm", value)}
          min={35}
          max={70}
          step={0.5}
          disabled={disabled}
        />
        <NumberInput
          label={messages.minHeightLabel}
          suffix={messages.pixelsSuffix}
          value={formValues.minHeightPx}
          onChange={(value) => onOptionChange("minHeightPx", value)}
          min={200}
          step={10}
          disabled={disabled}
        />
        <NumberInput
          label={messages.minWidthLabel}
          suffix={messages.pixelsSuffix}
          value={formValues.minWidthPx}
          onChange={(value) => onOptionChange("minWidthPx", value)}
          min={200}
          step={10}
          disabled={disabled}
        />
        <label className="flex items-start gap-3 rounded-lg border border-slate-200 bg-slate-50 p-3 text-sm text-slate-700 shadow-inner dark:border-slate-700 dark:bg-slate-800 dark:text-slate-200">
          <input
            type="checkbox"
            checked={formValues.saveDebug}
            onChange={onToggleDebug}
            disabled={disabled}
            className="mt-0.5 h-4 w-4 rounded border-slate-300 text-sky-500 focus:ring-sky-400 dark:border-slate-600 dark:bg-slate-900"
          />
          <span>
            <span className="font-semibold">{messages.saveDebugLabel}</span>
            <br />
            <span className="text-xs text-slate-500 dark:text-slate-400">
              {messages.saveDebugDescription}
            </span>
          </span>
        </label>
      </div>

      <div className="mt-4 rounded-xl border border-slate-200 bg-slate-50 p-4 dark:border-slate-700 dark:bg-slate-800">
        <button
          type="button"
          onClick={() => setShowAdvanced((current) => !current)}
          className="flex w-full items-center justify-between gap-2 text-sm font-semibold text-slate-700 transition hover:text-slate-900 focus:outline-none focus:ring-2 focus:ring-sky-400 dark:text-slate-200 dark:hover:text-white"
          aria-expanded={showAdvanced}
          aria-controls="advanced-options"
          aria-label={messages.ariaAdvancedToggle}
        >
          <span>{messages.advancedHeading}</span>
          <span className="text-xs font-normal text-slate-500 dark:text-slate-400">
            {messages.advancedSummary}
          </span>
        </button>
        {showAdvanced && (
          <div id="advanced-options" className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2">
            <NumberInput
              label={messages.targetRatioLabel}
              suffix={messages.ratioSuffix}
              value={formValues.targetWOverH}
              onChange={(value) => onOptionChange("targetWOverH", value)}
              step={0.01}
              min={0.5}
              max={1.5}
              disabled={disabled}
            />
            <NumberInput
              label={messages.topMarginLabel}
              value={formValues.topMarginRatio}
              onChange={(value) => onOptionChange("topMarginRatio", value)}
              step={0.01}
              min={0}
              max={0.5}
              disabled={disabled}
            />
            <NumberInput
              label={messages.bottomUpperLabel}
              value={formValues.bottomUpperRatio}
              onChange={(value) => onOptionChange("bottomUpperRatio", value)}
              step={0.01}
              min={0.5}
              max={1}
              disabled={disabled}
            />
            <NumberInput
              label={messages.maxCrownToChinLabel}
              suffix={messages.measurementUnitMm}
              value={formValues.maxCrownToChinMm}
              onChange={(value) => onOptionChange("maxCrownToChinMm", value)}
              step={0.5}
              min={10}
              max={60}
              disabled={disabled}
            />
            <NumberInput
              label={messages.minCrownToChinLabel}
              suffix={messages.measurementUnitMm}
              value={formValues.minCrownToChinMm}
              onChange={(value) => onOptionChange("minCrownToChinMm", value)}
              step={0.5}
              min={10}
              max={60}
              disabled={disabled}
            />
            <NumberInput
              label={messages.targetCrownToChinLabel}
              suffix={messages.measurementUnitMm}
              value={formValues.targetCrownToChinMm}
              onChange={(value) => onOptionChange("targetCrownToChinMm", value)}
              step={0.5}
              min={10}
              max={60}
              disabled={disabled}
            />
            <NumberInput
              label={messages.maxExtraPaddingLabel}
              suffix={messages.pixelsSuffix}
              value={formValues.maxExtraPaddingPx}
              onChange={(value) => onOptionChange("maxExtraPaddingPx", value)}
              step={10}
              min={0}
              max={2000}
              disabled={disabled}
            />
          </div>
        )}
      </div>

      <div className="mt-5 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="text-xs text-slate-500 dark:text-slate-400">{messages.processingTimeHint}</div>
        <div className="flex flex-wrap items-center gap-3">
          {status === "error" && (
            <button
              type="button"
              onClick={onRetry}
              className="rounded-md border border-rose-200 px-4 py-2 text-sm font-semibold text-rose-600 transition hover:border-rose-300 hover:text-rose-700 focus:outline-none focus:ring-4 focus:ring-rose-200 dark:border-rose-900 dark:text-rose-300 dark:hover:border-rose-700 dark:hover:text-rose-100 dark:focus:ring-rose-900"
            >
              {messages.retryButton}
            </button>
          )}
          <button
            type="submit"
            disabled={!canSubmit}
            className="rounded-md bg-sky-600 px-5 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-sky-700 focus:outline-none focus:ring-4 focus:ring-sky-200 disabled:cursor-not-allowed disabled:bg-slate-300 dark:bg-sky-500 dark:hover:bg-sky-400 dark:focus:ring-sky-900"
          >
            {messages.submitButton}
          </button>
        </div>
      </div>
      {!canSubmit && status !== "processing" && (
        <p className="mt-2 text-xs text-slate-500 dark:text-slate-400">{messages.submitDisabled}</p>
      )}
    </section>
  );
}

type NumberInputProps = {
  label: string;
  value: number;
  onChange: (value: number) => void;
  suffix?: string;
  step?: number;
  min?: number;
  max?: number;
  disabled?: boolean;
};

function NumberInput({ label, value, onChange, suffix, step, min, max, disabled }: NumberInputProps) {
  return (
    <label className="flex flex-col gap-1 text-sm font-medium text-slate-700 dark:text-slate-200">
      <span>{label}</span>
      <div className="relative flex items-center">
        <input
          type="number"
          inputMode="decimal"
          value={value}
          min={min}
          max={max}
          step={step}
          disabled={disabled}
          onChange={(event) => {
            const next = Number.parseFloat(event.target.value);
            onChange(Number.isFinite(next) ? next : value);
          }}
          className={classNames(
            "w-full rounded-md border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 shadow-sm focus:border-sky-500 focus:outline-none focus:ring-4 focus:ring-sky-200 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100 dark:focus:border-sky-400 dark:focus:ring-sky-900",
            disabled && "opacity-60",
          )}
        />
        {suffix && (
          <span className="pointer-events-none absolute right-2 text-xs text-slate-500 dark:text-slate-400">
            {suffix}
          </span>
        )}
      </div>
    </label>
  );
}
