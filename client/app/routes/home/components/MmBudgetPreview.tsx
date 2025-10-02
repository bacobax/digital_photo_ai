import classNames from "classnames";
import { useMemo } from "react";

type MmBudgetPreviewProps = {
  targetHeightMm: number;
  minTopMm: number;
  shoulderClearanceMm: number;
  minCrownToChinMm: number;
  maxCrownToChinMm: number;
  targetCrownToChinMm: number;
  unitLabel: string;
  heading: string;
  description: string;
  topLabel: string;
  faceLabel: string;
  bottomLabel: string;
  shoulderHint: string;
  totalLabel: string;
};

type MmBudgetGeometry = {
  topMm: number;
  faceMm: number;
  bottomMm: number;
  totalMm: number;
  topPercent: number;
  facePercent: number;
  bottomPercent: number;
  topLimited: boolean;
  bottomLimited: boolean;
  faceLimited: boolean;
};

export function MmBudgetPreview(props: MmBudgetPreviewProps) {
  const geometry = useMemo(() => computeBudgetGeometry(props), [props]);

  return (
    <div className="flex flex-col gap-3 rounded-xl border border-slate-200 bg-white/80 p-4 shadow-inner dark:border-slate-700 dark:bg-slate-900/60">
      <div className="text-sm font-semibold text-slate-700 dark:text-slate-200">{props.heading}</div>
      <div className="text-xs text-slate-500 dark:text-slate-400">{props.description}</div>
      <div className="mx-auto flex w-full max-w-[220px] flex-col items-center gap-4 sm:flex-row sm:items-start">
        <div className="relative flex h-48 w-24 flex-col overflow-hidden rounded-md border border-slate-300 dark:border-slate-700">
          <BudgetSegment
            percent={geometry.topPercent}
            label={props.topLabel}
            value={formatMm(geometry.topMm, props.unitLabel)}
            tone="top"
            limited={geometry.topLimited}
          />
          <BudgetSegment
            percent={geometry.facePercent}
            label={props.faceLabel}
            value={formatMm(geometry.faceMm, props.unitLabel)}
            tone="face"
            limited={geometry.faceLimited}
          />
          <BudgetSegment
            percent={geometry.bottomPercent}
            label={props.bottomLabel}
            value={formatMm(geometry.bottomMm, props.unitLabel)}
            tone="bottom"
            limited={geometry.bottomLimited}
          />
        </div>
        <div className="flex flex-1 flex-col gap-2 text-xs text-slate-600 dark:text-slate-400">
          <PreviewStat
            label={props.topLabel}
            value={formatMm(geometry.topMm, props.unitLabel)}
            limited={geometry.topLimited}
          />
          <PreviewStat
            label={props.faceLabel}
            value={formatMm(geometry.faceMm, props.unitLabel)}
            limited={geometry.faceLimited}
          />
          <PreviewStat
            label={props.bottomLabel}
            value={formatMm(geometry.bottomMm, props.unitLabel)}
            limited={geometry.bottomLimited}
          />
          <PreviewStat
            label={props.totalLabel}
            value={formatMm(geometry.totalMm, props.unitLabel)}
          />
          <div className="rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-[0.7rem] font-medium text-amber-700 shadow-sm dark:border-amber-800 dark:bg-amber-950/50 dark:text-amber-200">
            {props.shoulderHint.replace("{clearance}", formatMm(props.shoulderClearanceMm, props.unitLabel))}
          </div>
        </div>
      </div>
    </div>
  );
}

type BudgetSegmentProps = {
  percent: number;
  label: string;
  value: string;
  tone: "top" | "face" | "bottom";
  limited?: boolean;
};

function BudgetSegment({ percent, label, value, tone, limited }: BudgetSegmentProps) {
  return (
    <div
      className={classNames(
        "flex flex-col items-center justify-center gap-1 border-b border-slate-200 px-2 text-center text-[0.7rem] font-semibold uppercase tracking-wide dark:border-slate-700",
        tone === "top" && "bg-sky-50 text-sky-700 dark:bg-sky-950/40 dark:text-sky-200",
        tone === "face" && "bg-emerald-50 text-emerald-700 dark:bg-emerald-950/40 dark:text-emerald-200",
        tone === "bottom" && "bg-indigo-50 text-indigo-700 dark:bg-indigo-950/40 dark:text-indigo-200",
        limited && "outline outline-2 outline-dashed outline-amber-400/70",
      )}
      style={{ flexGrow: Math.max(percent, 0.1), flexBasis: 0, minHeight: "2.75rem" }}
    >
      <span>{label}</span>
      <span className="text-[0.65rem] font-normal normal-case text-slate-600 dark:text-slate-300">{value}</span>
      {limited && (
        <span className="text-[0.6rem] font-semibold uppercase text-amber-600 dark:text-amber-300">Min</span>
      )}
    </div>
  );
}

type PreviewStatProps = {
  label: string;
  value: string;
  limited?: boolean;
};

function PreviewStat({ label, value, limited }: PreviewStatProps) {
  return (
    <div
      className={classNames(
        "flex items-center justify-between gap-3 rounded-md border border-slate-200 bg-white/70 px-3 py-2 text-[0.75rem] font-medium text-slate-600 shadow-sm dark:border-slate-700 dark:bg-slate-900/50 dark:text-slate-300",
        limited && "border-amber-300 text-amber-700 dark:border-amber-700 dark:text-amber-200",
      )}
    >
      <span>{label}</span>
      <span className="text-sm font-semibold text-slate-800 dark:text-slate-100">{value}</span>
    </div>
  );
}

function formatMm(value: number, unitLabel: string) {
  return `${value.toFixed(1)} ${unitLabel}`;
}

function computeBudgetGeometry({
  targetHeightMm,
  minTopMm,
  minCrownToChinMm,
  maxCrownToChinMm,
  targetCrownToChinMm,
}: MmBudgetPreviewProps): MmBudgetGeometry {
  const safeTargetHeight = Math.max(targetHeightMm, 1);
  const requestedTop = Math.max(0, minTopMm);
  const crownMin = Math.max(1, minCrownToChinMm);
  const crownMax = Math.max(crownMin, maxCrownToChinMm);

  const faceMm = clamp(targetCrownToChinMm, crownMin, crownMax);
  const topMm = Math.min(requestedTop, safeTargetHeight);
  const bottomMm = Math.max(0, safeTargetHeight - topMm - faceMm);

  const totalMm = topMm + faceMm + bottomMm;
  const normaliser = totalMm > 0 ? totalMm : 1;

  return {
    topMm,
    faceMm,
    bottomMm,
    totalMm,
    topPercent: (topMm / normaliser) * 100,
    facePercent: (faceMm / normaliser) * 100,
    bottomPercent: (bottomMm / normaliser) * 100,
    topLimited: topMm + 0.05 < requestedTop,
    bottomLimited: bottomMm <= 0.05,
    faceLimited: Math.abs(faceMm - crownMin) <= 0.05 || Math.abs(faceMm - crownMax) <= 0.05,
  };
}

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
}

