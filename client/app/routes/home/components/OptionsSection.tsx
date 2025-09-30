import type { ChangeEvent } from "react";
import { useEffect, useMemo, useState } from "react";
import classNames from "classnames";

import type { OptionsSectionProps } from "../types";
import { CropPreview } from "./CropPreview";

export function OptionsSection({
  messages,
  formValues,
  onOptionChange,
  onPipelineChange,
  onToggleDebug,
  status,
  canSubmit,
  onRetry,
}: OptionsSectionProps) {
  const [showAdvanced, setShowAdvanced] = useState(false);

  const disabled = status === "processing";
  const isClosedForm = formValues.pipeline === "closed_form";

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
          value={formValues.target_height_mm}
          onChange={(value) => onOptionChange("target_height_mm", value)}
          min={35}
          max={70}
          step={0.1}
          disabled={disabled}
        />
        <NumberInput
          label={messages.minHeightLabel}
          suffix={messages.pixelsSuffix}
          value={formValues.min_height_px}
          onChange={(value) => onOptionChange("min_height_px", value)}
          min={200}
          step={2}
          disabled={disabled}
        />
        <NumberInput
          label={messages.minWidthLabel}
          suffix={messages.pixelsSuffix}
          value={formValues.min_width_px}
          onChange={(value) => onOptionChange("min_width_px", value)}
          min={200}
          step={2}
          disabled={disabled}
        />
        <label className="flex items-start gap-3 rounded-lg border border-slate-200 bg-slate-50 p-3 text-sm text-slate-700 shadow-inner dark:border-slate-700 dark:bg-slate-800 dark:text-slate-200">
          <input
            type="checkbox"
            checked={formValues.save_debug}
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

      <fieldset className="mt-4 rounded-xl border border-slate-200 bg-slate-50 p-4 dark:border-slate-700 dark:bg-slate-800">
        <legend className="text-sm font-semibold text-slate-700 dark:text-slate-200">
          {messages.pipelineHeading}
        </legend>
        <div className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-2">
          <PipelineOptionCard
            label={messages.pipelineClosedFormLabel}
            description={messages.pipelineClosedFormDescription}
            selected={isClosedForm}
            onSelect={() => onPipelineChange("closed_form")}
            disabled={disabled}
          />
          <PipelineOptionCard
            label={messages.pipelineLegacyLabel}
            description={messages.pipelineLegacyDescription}
            selected={!isClosedForm}
            onSelect={() => onPipelineChange("legacy")}
            disabled={disabled}
          />
        </div>
      </fieldset>

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
            <RatioInput
              label={messages.targetRatioLabel}
              value={formValues.target_w_over_h}
              onChange={(value) => onOptionChange("target_w_over_h", value)}
              min={0.5}
              max={1.5}
              disabled={disabled}
            />
            <PercentageSlider
              label={messages.resizeScalingLabel}
              value={formValues.resize_scaling}
              onChange={(value) => onOptionChange("resize_scaling", value)}
              minPercent={0}
              maxPercent={100}
              disabled={disabled}
            />
            {isClosedForm ? (
              <>
                <NumberInput
                  label={messages.minTopMmLabel}
                  suffix={messages.measurementUnitMm}
                  value={formValues.min_top_mm}
                  onChange={(value) => onOptionChange("min_top_mm", value)}
                  step={0.5}
                  min={0}
                  disabled={disabled}
                />
                <NumberInput
                  label={messages.minBottomMmLabel}
                  suffix={messages.measurementUnitMm}
                  value={formValues.min_bottom_mm}
                  onChange={(value) => onOptionChange("min_bottom_mm", value)}
                  step={0.5}
                  min={0}
                  disabled={disabled}
                />
                <NumberInput
                  label={messages.shoulderClearanceLabel}
                  suffix={messages.measurementUnitMm}
                  value={formValues.shoulder_clearance_mm}
                  onChange={(value) => onOptionChange("shoulder_clearance_mm", value)}
                  step={0.5}
                  min={0}
                  disabled={disabled}
                />
              </>
            ) : (
              <>
                <PercentageSlider
                  label={messages.topMarginLabel}
                  value={formValues.top_margin_ratio}
                  onChange={(value) => onOptionChange("top_margin_ratio", value)}
                  minPercent={0}
                  maxPercent={50}
                  disabled={disabled}
                />
                <PercentageSlider
                  label={messages.bottomUpperLabel}
                  value={formValues.bottom_upper_ratio}
                  onChange={(value) => onOptionChange("bottom_upper_ratio", value)}
                  minPercent={50}
                  maxPercent={100}
                  disabled={disabled}
                />
                <div className="sm:col-span-2">
                  <CropPreview
                    widthHeightRatio={formValues.target_w_over_h}
                    topMarginRatio={formValues.top_margin_ratio}
                    lowerFaceRatio={formValues.bottom_upper_ratio}
                  />
                </div>
              </>
            )}
            <NumberInput
              label={messages.maxCrownToChinLabel}
              suffix={messages.measurementUnitMm}
              value={formValues.max_crown_to_chin_mm}
              onChange={(value) => onOptionChange("max_crown_to_chin_mm", value)}
              step={0.5}
              min={10}
              max={60}
              disabled={disabled}
            />
            <NumberInput
              label={messages.minCrownToChinLabel}
              suffix={messages.measurementUnitMm}
              value={formValues.min_crown_to_chin_mm}
              onChange={(value) => onOptionChange("min_crown_to_chin_mm", value)}
              step={0.5}
              min={10}
              max={60}
              disabled={disabled}
            />
            <NumberInput
              label={messages.targetCrownToChinLabel}
              suffix={messages.measurementUnitMm}
              value={formValues.target_crown_to_chin_mm}
              onChange={(value) => onOptionChange("target_crown_to_chin_mm", value)}
              step={0.5}
              min={10}
              max={60}
              disabled={disabled}
            />
            <NumberInput
              label={messages.maxExtraPaddingLabel}
              suffix={messages.pixelsSuffix}
              value={formValues.max_extra_padding_px}
              onChange={(value) => onOptionChange("max_extra_padding_px", value)}
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

type RatioInputProps = {
  label: string;
  value: number;
  onChange: (value: number) => void;
  disabled?: boolean;
  min?: number;
  max?: number;
};

function RatioInput({ label, value, onChange, disabled, min, max }: RatioInputProps) {
  const initialParts = useMemo(() => fractionFromRatio(value), [value]);
  const [numerator, setNumerator] = useState<number>(initialParts.numerator);
  const [denominator, setDenominator] = useState<number>(initialParts.denominator);

  const emitChange = (nextNumerator: number, nextDenominator: number) => {
    if (!Number.isFinite(nextNumerator) || !Number.isFinite(nextDenominator) || nextDenominator <= 0) {
      return;
    }
    const nextRatio = nextNumerator / nextDenominator;
    if (!Number.isFinite(nextRatio)) {
      return;
    }
    const constrained = clamp(nextRatio, min, max);
    onChange(constrained);
  };

  const handleNumeratorChange = (raw: string) => {
    const parsed = Number.parseFloat(raw);
    const next = Number.isFinite(parsed) ? Math.max(Math.round(parsed), 0) : numerator;
    setNumerator(next);
    emitChange(next, denominator);
  };

  const handleDenominatorChange = (raw: string) => {
    const parsed = Number.parseFloat(raw);
    const next = Number.isFinite(parsed) ? Math.max(Math.round(parsed), 1) : denominator;
    setDenominator(next);
    emitChange(numerator, next);
  };

  return (
    <label className="flex flex-col gap-1 text-sm font-medium text-slate-700 dark:text-slate-200">
      <span>{label}</span>
      <div className="flex items-center gap-2">
        <input
          type="number"
          inputMode="numeric"
          value={formatRatioPart(numerator)}
          onChange={(event) => handleNumeratorChange(event.target.value)}
          disabled={disabled}
          min={0}
          className={classNames(
            "w-20 rounded-md border border-slate-300 bg-white px-2 py-2 text-sm text-slate-900 shadow-sm focus:border-sky-500 focus:outline-none focus:ring-4 focus:ring-sky-200 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100 dark:focus:border-sky-400 dark:focus:ring-sky-900",
            disabled && "opacity-60",
          )}
        />
        <span className="text-base font-semibold text-slate-500 dark:text-slate-400">:</span>
        <input
          type="number"
          inputMode="numeric"
          value={formatRatioPart(denominator)}
          onChange={(event) => handleDenominatorChange(event.target.value)}
          disabled={disabled}
          min={1}
          className={classNames(
            "w-20 rounded-md border border-slate-300 bg-white px-2 py-2 text-sm text-slate-900 shadow-sm focus:border-sky-500 focus:outline-none focus:ring-4 focus:ring-sky-200 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-100 dark:focus:border-sky-400 dark:focus:ring-sky-900",
            disabled && "opacity-60",
          )}
        />
      </div>
    </label>
  );
}

type PercentageSliderProps = {
  label: string;
  value: number;
  onChange: (value: number) => void;
  minPercent: number;
  maxPercent: number;
  disabled?: boolean;
};

function PercentageSlider({ label, value, onChange, minPercent, maxPercent, disabled }: PercentageSliderProps) {
  const sliderValue = clamp(Math.round(value * 100), minPercent, maxPercent);

  const handleChange = (event: ChangeEvent<HTMLInputElement>) => {
    const nextPercent = Number.parseInt(event.target.value, 10);
    if (!Number.isFinite(nextPercent)) {
      return;
    }
    const clamped = clamp(nextPercent, minPercent, maxPercent);
    onChange(clamped / 100);
  };

  return (
    <label className="flex flex-col gap-2 text-sm font-medium text-slate-700 dark:text-slate-200">
      <span className="flex items-center justify-between">
        <span>{label}</span>
        <span className="text-xs font-semibold text-slate-500 dark:text-slate-400">{sliderValue}%</span>
      </span>
      <input
        type="range"
        min={minPercent}
        max={maxPercent}
        value={sliderValue}
        step={1}
        disabled={disabled}
        onChange={handleChange}
        className="h-2 w-full cursor-pointer appearance-none rounded-full bg-slate-200 accent-sky-500 focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-500 dark:bg-slate-700"
      />
    </label>
  );
}

type PipelineOptionCardProps = {
  label: string;
  description: string;
  selected: boolean;
  onSelect: () => void;
  disabled: boolean;
};

function PipelineOptionCard({ label, description, selected, onSelect, disabled }: PipelineOptionCardProps) {
  return (
    <button
      type="button"
      onClick={onSelect}
      disabled={disabled}
      aria-pressed={selected}
      className={classNames(
        "flex h-full flex-col gap-1 rounded-lg border border-slate-200 bg-white p-3 text-left shadow-sm transition hover:border-sky-400 focus:outline-none focus:ring-2 focus:ring-sky-400 dark:border-slate-700 dark:bg-slate-900 dark:hover:border-sky-500 dark:focus:ring-sky-800",
        selected && "border-sky-500 ring-2 ring-sky-200 dark:border-sky-400 dark:ring-sky-900",
        disabled && "cursor-not-allowed opacity-60 hover:border-slate-200 dark:hover:border-slate-700",
      )}
    >
      <span className="text-sm font-semibold text-slate-800 dark:text-slate-100">{label}</span>
      <span className="text-xs text-slate-500 dark:text-slate-400">{description}</span>
    </button>
  );
}

function clamp(value: number, minValue?: number, maxValue?: number) {
  let result = value;
  if (typeof minValue === "number") {
    result = Math.max(result, minValue);
  }
  if (typeof maxValue === "number") {
    result = Math.min(result, maxValue);
  }
  return result;
}

function fractionFromRatio(ratio: number) {
  if (!Number.isFinite(ratio) || ratio <= 0) {
    return { numerator: 1, denominator: 1 };
  }
  const maxDenominator = 50;
  let bestNumerator = 1;
  let bestDenominator = 1;
  let bestError = Number.POSITIVE_INFINITY;

  for (let denominator = 1; denominator <= maxDenominator; denominator += 1) {
    const numerator = Math.max(1, Math.round(ratio * denominator));
    const approximation = numerator / denominator;
    const error = Math.abs(approximation - ratio);
    if (error < bestError) {
      bestError = error;
      bestNumerator = numerator;
      bestDenominator = denominator;
    }
    if (bestError === 0) {
      break;
    }
  }
  return { numerator: bestNumerator, denominator: bestDenominator };
}

function formatRatioPart(value: number) {
  if (!Number.isFinite(value)) {
    return "";
  }
  const rounded = Math.round(value * 1000) / 1000;
  return Number.isInteger(rounded) ? `${rounded}` : rounded.toString();
}
