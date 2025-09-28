import classNames from "classnames";
import { useMemo } from "react";

const MIN_RATIO = 0.3;
const MAX_RATIO = 2.0;

export type CropPreviewProps = {
  widthHeightRatio: number;
  topMarginRatio: number;
  lowerFaceRatio: number;
};

export function CropPreview({ widthHeightRatio, topMarginRatio, lowerFaceRatio }: CropPreviewProps) {
  const geometry = useMemo(
    () => computeGeometry(widthHeightRatio, topMarginRatio, lowerFaceRatio),
    [lowerFaceRatio, topMarginRatio, widthHeightRatio],
  );

  return (
    <div className="flex flex-col gap-3 rounded-xl border border-slate-200 bg-white/80 p-4 shadow-inner dark:border-slate-700 dark:bg-slate-900/60">
      <div className="text-sm font-semibold text-slate-700 dark:text-slate-200">Live framing preview</div>
      <div className="text-xs text-slate-500 dark:text-slate-400">
        Adjust the controls to see how the top margin, lower face ratio, and width/height ratio change the crop box and face placement.
      </div>
      <div className="relative mx-auto w-full max-w-[180px] rounded-lg border border-slate-300 bg-white dark:border-slate-700 dark:bg-slate-900/70">
        <div
          className="relative w-full"
          style={{ paddingBottom: `${geometry.paddingBottomPercent}%` }}
        >
          <div className="absolute inset-0">
            <HorizontalGuide topPercent={0} label="Top" />
            <HorizontalGuide
              topPercent={geometry.crownPercent}
              label="Crown"
              tone="primary"
            />
            <HorizontalGuide
              topPercent={geometry.chinPercent}
              label="Chin"
              tone="accent"
            />
            <HorizontalGuide topPercent={100} label="Bottom" />
            <FaceEllipse geometry={geometry} />
            <ShoulderShape geometry={geometry} />

          </div>
        </div>
      </div>
      <div className="grid gap-2 text-xs text-slate-600 dark:text-slate-400 sm:grid-cols-3">
        <Stat label="Aspect" value={`${geometry.safeRatio.toFixed(2)} : 1`} />
        <Stat label="Top margin" value={`${(geometry.topMarginPercent).toFixed(0)}%`} />
        <Stat label="Lower padding" value={`${geometry.bottomMarginPercent.toFixed(0)}%`} />
      </div>
    </div>
  );
}

function HorizontalGuide({
  topPercent,
  label,
  tone,
}: {
  topPercent: number;
  label: string;
  tone?: "primary" | "accent";
}) {
  return (
    <div
      className={classNames(
        "absolute left-0 flex w-full items-center gap-2 px-2",
        tone === "primary" && "text-sky-500",
        tone === "accent" && "text-emerald-500",
        !tone && "text-slate-400",
      )}
      style={{ top: `${topPercent}%`, transform: "translateY(-50%)" }}
    >
      <div
        className={classNames(
          "h-px flex-1 bg-slate-300 dark:bg-slate-600",
          tone === "primary" && "bg-sky-400 dark:bg-sky-500",
          tone === "accent" && "bg-emerald-400 dark:bg-emerald-500",
        )}
      />
      <span className="rounded-full bg-white/70 px-2 py-0.5 text-[1rem] font-semibold uppercase tracking-wide shadow dark:bg-slate-900/70 ">
        {label}
      </span>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-slate-200 bg-white/60 p-2 text-center shadow-sm dark:border-slate-700 dark:bg-slate-900/60">
      <div className="text-[0.65rem] uppercase tracking-wide text-slate-500 dark:text-slate-400">
        {label}
      </div>
      <div className="text-sm font-semibold text-slate-700 dark:text-slate-100">{value}</div>
    </div>
  );
}

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
}

type FaceGeometry = ReturnType<typeof computeGeometry>;

function FaceEllipse({ geometry }: { geometry: FaceGeometry }) {
  const {
    crownPercent,
    ellipseHeightPercent,
    ellipseWidthPercent,
    faceHeightPercent,
  } = geometry;

  const ellipseStyle = {
    height: `${ellipseHeightPercent}%`,
    width: `${ellipseWidthPercent}%`,
    background: "radial-gradient(circle at 50% 30%, rgba(252, 211, 77, 0.6), rgba(251, 191, 36, 0.25))",
    boxShadow: "0 10px 25px rgba(30, 41, 59, 0.15)",
  } as const;

  return (
    <div
      className="absolute flex flex-col items-center"
      style={{ top: `${crownPercent}%`, left: "50%", transform: "translate(-50%, 0)" }}
    >
      <div
        className="rounded-full border border-amber-400/60 dark:border-amber-300/60"
        style={ellipseStyle}
      />
    </div>
  );
}

function ShoulderShape({ geometry }: { geometry: FaceGeometry }) {
  const { chinPercent, ellipseWidthPercent, shoulderHeightPercent } = geometry;
  if (shoulderHeightPercent <= 0) {
    return null;
  }
  const shoulderWidthPercent = Math.min(ellipseWidthPercent * 1.6, 98);
  const shoulderStyle = {
    width: `${shoulderWidthPercent}%`,
    height: `${shoulderHeightPercent}%`,
    borderBottomLeftRadius: "50% 100%",
    borderBottomRightRadius: "50% 100%",
    background: "linear-gradient(180deg, rgba(148, 163, 184, 0.25), rgba(71, 85, 105, 0.4))",
  } as const;
  return (
    <div
      className="absolute flex justify-center"
      style={{ top: `${chinPercent}%`, left: "50%", transform: "translate(-50%, 0)" }}
    >
      <div className="overflow-hidden" style={shoulderStyle} />
    </div>
  );
}

function computeGeometry(widthHeightRatio: number, topMarginRatio: number, lowerFaceRatio: number) {
  const safeRatio = clamp(widthHeightRatio, MIN_RATIO, MAX_RATIO);
  const safeTopMargin = Math.max(topMarginRatio, 0);
  const safeLowerRatio = clamp(lowerFaceRatio, 0.05, 0.95);

  const faceHeight = 1;
  const topMargin = safeTopMargin * faceHeight;
  const crownDistance = topMargin;
  const chinDistance = topMargin + faceHeight;
  const totalHeight = Math.max(chinDistance / safeLowerRatio, chinDistance + 1e-6);
  const bottomMargin = Math.max(totalHeight - chinDistance, 0);

  const crownPercent = (crownDistance / totalHeight) * 100;
  const chinPercent = (chinDistance / totalHeight) * 100;
  const topMarginPercent = crownPercent;
  const bottomMarginPercent = (bottomMargin / totalHeight) * 100;
  const faceHeightPercent = (faceHeight / totalHeight) * 100;

  const ellipseHeightPercent = faceHeightPercent;
  const ellipseWidthPercent = Math.min(faceHeightPercent * safeRatio, 95);
  const shoulderHeightPercent = Math.max(bottomMarginPercent, 0);
  const shoulderTopPercent = Math.min(100, chinPercent);
  const paddingBottomPercent = (1 / safeRatio) * 100;

  return {
    safeRatio,
    paddingBottomPercent,
    crownPercent,
    chinPercent,
    topMarginPercent,
    bottomMarginPercent,
    faceHeightPercent,
    ellipseHeightPercent,
    ellipseWidthPercent,
    shoulderHeightPercent,
    shoulderTopPercent,
    topMargin,
    bottomMargin,
    faceHeight,
  };
}
