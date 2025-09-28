import classNames from "classnames";
import { useEffect, useMemo, useState } from "react";

import type { AnnotatedPreviewProps, FaceMarkerKey, FaceSpanKey } from "../types";

const MARKER_COLORS: Record<FaceMarkerKey, string> = {
  crown: "#ef4444",
  forehead: "#f97316",
  chin: "#22c55e",
};

const SPAN_COLORS: Record<FaceSpanKey, string> = {
  crownToChin: MARKER_COLORS.crown,
  foreheadToChin: MARKER_COLORS.forehead,
};

export function AnnotatedPreview({ image, markers, spans, messages }: AnnotatedPreviewProps) {
  const [naturalSize, setNaturalSize] = useState<{ width: number; height: number } | null>(null);
  const [hoveredSpanKey, setHoveredSpanKey] = useState<FaceSpanKey | null>(null);
  const [lockedSpanKey, setLockedSpanKey] = useState<FaceSpanKey | null>(null);

  useEffect(() => {
    setNaturalSize(null);
    setHoveredSpanKey(null);
    setLockedSpanKey(null);
  }, [image?.url]);

  useEffect(() => {
    if (hoveredSpanKey && !spans.some((span) => span.key === hoveredSpanKey)) {
      setHoveredSpanKey(null);
    }
    if (lockedSpanKey && !spans.some((span) => span.key === lockedSpanKey)) {
      setLockedSpanKey(null);
    }
  }, [hoveredSpanKey, lockedSpanKey, spans]);

  const positionedMarkers = useMemo(() => {
    if (!naturalSize) {
      return [];
    }
    const clampedHeight = Math.max(naturalSize.height, 1);

    return markers
      .map((marker) => {
        if (marker.distanceFromBottomPx === null) {
          return null;
        }
        const percent = 100 - (marker.distanceFromBottomPx / clampedHeight) * 100;
        const positionPercent = Math.min(100, Math.max(0, percent));
        return {
          ...marker,
          positionPercent,
        };
      })
      .filter((marker): marker is (typeof markers)[number] & { positionPercent: number } => marker !== null);
  }, [markers, naturalSize]);

  const markerMap = useMemo(() => {
    const entries = positionedMarkers.map((marker) => [marker.key, marker] as const);
    return new Map(entries);
  }, [positionedMarkers]);

  const positionedSpans = useMemo(() => {
    if (!naturalSize || markerMap.size === 0) {
      return [];
    }

    return spans
      .map((span) => {
        const startMarker = markerMap.get(span.startKey);
        const endMarker = markerMap.get(span.endKey);
        if (!startMarker || !endMarker) {
          return null;
        }
        const rawTop = Math.min(startMarker.positionPercent, endMarker.positionPercent);
        const rawBottom = Math.max(startMarker.positionPercent, endMarker.positionPercent);
        const clampedTop = Math.min(100, Math.max(0, rawTop));
        const clampedBottom = Math.min(100, Math.max(0, rawBottom));
        const rawHeight = clampedBottom - clampedTop;
        const heightPercent = Math.max(rawHeight, 0.5);
        const topPercent = Math.min(clampedTop, 100 - heightPercent);
        const bottomPercent = Math.min(100, topPercent + heightPercent);
        const color = SPAN_COLORS[span.key] ?? MARKER_COLORS.chin;
        const spanHeight = bottomPercent - topPercent;
        const midpointPercent = topPercent + spanHeight / 2;
        const relativeOffset = spanHeight > 0 ? ((midpointPercent - topPercent) / spanHeight) * 100 : 50;
        const labelOffsetPercent = Math.min(Math.max(relativeOffset, 0), 100);
        return {
          span,
          topPercent,
          bottomPercent,
          heightPercent: spanHeight,
          color,
          midpointPercent,
          labelOffsetPercent,
        };
      })
      .filter((entry): entry is {
        span: (typeof spans)[number];
        topPercent: number;
        bottomPercent: number;
        heightPercent: number;
        color: string;
        midpointPercent: number;
        labelOffsetPercent: number;
      } => entry !== null);
  }, [markerMap, naturalSize, spans]);

  const measurementUnitPx = messages.measurementUnitPx;
  const measurementUnitMm = messages.measurementUnitMm;

  return (
    <div className="flex flex-col gap-4">
      <div className="relative mx-auto w-full max-w-xs overflow-hidden rounded-xl border border-slate-200 bg-slate-100 shadow-inner dark:border-slate-700 dark:bg-slate-800">
        {image ? (
          <>
            <img
              src={image.url}
              alt={messages.annotatedAlt}
              className="block w-full object-cover"
              onLoad={(event) => {
                const target = event.currentTarget;
                setNaturalSize({ width: target.naturalWidth, height: target.naturalHeight });
              }}
            />
            {naturalSize && positionedMarkers.length > 0 && (
              <div className="absolute inset-0 pointer-events-none">
                {positionedMarkers.map((marker) => {
                  const color = MARKER_COLORS[marker.key] ?? MARKER_COLORS.chin;
                  const placeLabelBelow = marker.positionPercent < 16;
                  return (
                    <div
                      key={marker.key}
                      className="pointer-events-none absolute inset-x-0"
                      style={{ top: `${marker.positionPercent}%` }}
                    >
                      <div className="relative">
                        <div
                          className="h-[2px] w-full"
                          style={{ backgroundColor: color, boxShadow: `0 0 6px ${color}55` }}
                        />
                        <div
                          className={classNames(
                            "absolute left-2 inline-flex items-center gap-2 rounded-full px-3 py-1 text-xs font-semibold text-slate-900 shadow-lg ring-1 ring-black/10 backdrop-blur",
                            "dark:text-slate-900 dark:ring-white/10",
                            placeLabelBelow
                              ? "translate-y-1 bg-white/90 dark:bg-slate-200/90"
                              : "-translate-y-full -translate-y-2 bg-white/90 dark:bg-slate-200/90",
                          )}
                          title={
                            marker.distanceFromBottomPx !== null
                              ? `${marker.label} - ${Math.round(marker.distanceFromBottomPx)} ${messages.measurementUnitPx}`
                              : marker.label
                          }
                        >
                          <span
                            className="inline-flex h-2.5 w-2.5 rounded-full"
                            style={{ backgroundColor: color }}
                            aria-hidden="true"
                          />
                          {marker.label}
                        </div>
                      </div>
                    </div>
                  );
                })}

                {positionedSpans.map(
                  ({ span, topPercent, heightPercent, color, labelOffsetPercent }) => {
                    const isHovered = hoveredSpanKey === span.key;
                    const isLocked = lockedSpanKey === span.key;
                    const isVisible = isHovered || isLocked;
                    const measurementParts: string[] = [];
                    if (span.pixels !== null) {
                      measurementParts.push(`${Math.round(span.pixels)} ${measurementUnitPx}`);
                    }
                    if (span.millimeters !== null) {
                      measurementParts.push(`${span.millimeters.toFixed(2)} ${measurementUnitMm}`);
                    }
                    const ariaLabel = measurementParts.length > 0
                      ? `${span.label}. ${measurementParts.join(" / ")}`
                      : span.label;

                    return (
                      <div
                        key={span.key}
                        className="absolute right-2 flex w-20 justify-end"
                        style={{ top: `${topPercent}%`, height: `${heightPercent}%` }}
                      >
                        <button
                          type="button"
                          aria-label={ariaLabel}
                          aria-pressed={isLocked}
                          onPointerEnter={() => setHoveredSpanKey(span.key)}
                          onPointerLeave={() => setHoveredSpanKey(null)}
                          onFocus={() => setHoveredSpanKey(span.key)}
                          onBlur={() => setHoveredSpanKey(null)}
                          onClick={() =>
                            setLockedSpanKey((current) => (current === span.key ? null : span.key))
                          }
                          className="group relative h-full w-full cursor-pointer select-none rounded-lg focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-500 pointer-events-auto"
                          style={{ background: "none" }}
                        >
                          <span className="pointer-events-none absolute inset-0">
                            <div
                              className="absolute right-6 top-0 h-full w-[3px] rounded-full"
                              style={{ backgroundColor: `${color}dd` }}
                            />
                            <div
                              className="absolute right-[18px] h-3 w-3 -translate-x-1/2 rounded-full"
                              style={{ top: "0", backgroundColor: color }}
                              aria-hidden="true"
                            />
                            <div
                              className="absolute right-[18px] h-3 w-3 -translate-x-1/2 rounded-full"
                              style={{ bottom: "0", backgroundColor: color }}
                              aria-hidden="true"
                            />
                          </span>

                          <span
                            className={classNames(
                              "pointer-events-none absolute right-0 flex max-w-[150px] -translate-y-1/2 flex-col gap-0 rounded-lg bg-white/95 px-3 py-2 text-[0.65rem] font-medium text-slate-900 shadow-lg ring-1 ring-black/5 transition-all duration-150 dark:bg-slate-900/90 dark:text-slate-100 dark:ring-white/10",
                              isVisible ? "translate-x-3 opacity-100" : "translate-x-1 opacity-0",
                            )}
                            style={{ top: `${labelOffsetPercent}%` }}
                          >
                            <span>{span.label}</span>
                            <span className="text-[0.6rem] font-normal text-slate-500 dark:text-slate-300">
                              {span.pixels !== null && `${Math.round(span.pixels)} ${measurementUnitPx}`}
                              {span.pixels !== null && span.millimeters !== null && " / "}
                              {span.millimeters !== null && `${span.millimeters.toFixed(2)} ${measurementUnitMm}`}
                            </span>
                          </span>
                        </button>
                      </div>
                    );
                  },
                )}
              </div>
            )}
          </>
        ) : (
          <div className="flex aspect-[35/45] items-center justify-center text-sm text-slate-500 dark:text-slate-300">
            {messages.loadingPlaceholder}
          </div>
        )}
      </div>

      <div className="space-y-3">
        <h4 className="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
          {messages.overlayLegendHeading}
        </h4>
        {markers.length > 0 ? (
          <div className="flex flex-col gap-2">
            {markers.map((marker) => {
              const color = MARKER_COLORS[marker.key] ?? MARKER_COLORS.chin;
              return (
                <div
                  key={`legend-${marker.key}`}
                  className="flex items-center justify-between rounded-lg border border-slate-200 bg-white/80 px-3 py-2 text-sm shadow-sm dark:border-slate-700 dark:bg-slate-900/70"
                >
                  <span className="flex items-center gap-2 font-medium text-slate-700 dark:text-slate-200">
                    <span className="inline-flex h-2.5 w-2.5 rounded-full" style={{ backgroundColor: color }} aria-hidden="true" />
                    {marker.label}
                  </span>
                  <span className="text-xs text-slate-500 dark:text-slate-400">
                    {marker.distanceFromBottomPx !== null && `${Math.round(marker.distanceFromBottomPx)} ${messages.measurementUnitPx}`}
                    {marker.distanceFromBottomPx !== null && marker.distanceFromBottomMm !== null && " / "}
                    {marker.distanceFromBottomMm !== null && `${marker.distanceFromBottomMm.toFixed(1)} ${messages.measurementUnitMm}`}
                  </span>
                </div>
              );
            })}
          </div>
        ) : (
          <p className="text-sm text-slate-500 dark:text-slate-300">{messages.metadataMissing}</p>
        )}

        {spans.length > 0 && (
          <div className="flex flex-col gap-2">
            {spans.map((span) => {
              const color = SPAN_COLORS[span.key] ?? MARKER_COLORS.chin;
              return (
                <div
                  key={`span-${span.key}`}
                  className="flex items-center justify-between rounded-lg border border-slate-200 bg-white/80 px-3 py-2 text-sm shadow-sm dark:border-slate-700 dark:bg-slate-900/70"
                >
                  <span className="flex items-center gap-2 font-medium text-slate-700 dark:text-slate-200">
                    <span className="inline-flex h-2.5 w-2.5 rounded-full" style={{ backgroundColor: color }} aria-hidden="true" />
                    {span.label}
                  </span>
                  <span className="text-xs text-slate-500 dark:text-slate-400">
                    {span.pixels !== null && `${Math.round(span.pixels)} ${messages.measurementUnitPx}`}
                    {span.pixels !== null && span.millimeters !== null && " / "}
                    {span.millimeters !== null && `${span.millimeters.toFixed(2)} ${messages.measurementUnitMm}`}
                  </span>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
