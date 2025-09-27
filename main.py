import argparse
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from functions import (
    adjust_box_to_ratio,
    expand_box_to_hair_top,
    extend_box_downwards,
    top_of_hair_y_debug,
)

Box = Tuple[float, float, float, float]


@dataclass
class FacePipelineItem:
    original_box: Box
    top_margin_box: Optional[Box] = None
    bottom_scaled_box: Optional[Box] = None
    final_box: Optional[Box] = None
    chin_y_abs: Optional[float] = None
    top_face_y_abs: Optional[float] = None
    original_bottom_y_abs: Optional[float] = None
    hair_top_y_abs: Optional[float] = None
    crown_y_abs: Optional[float] = None
    chin_y_local: Optional[float] = None
    top_face_y_local: Optional[float] = None
    hair_top_y_local: Optional[float] = None
    crown_y_local: Optional[float] = None
    original_bottom_y_local: Optional[float] = None
    final_crop: Optional[np.ndarray] = None
    debug: Dict[str, np.ndarray] = field(default_factory=dict)
    top_margin_ratio_achieved: Optional[float] = None
    upper_ratio_achieved: Optional[float] = None


@dataclass
class RunParameters:
    target_w_over_h: float = 7.0 / 9.0
    top_margin_ratio: float = 0.10
    bottom_upper_ratio: float = 0.80
    min_width_px: int = 420
    min_height_px: int = 540
    target_height_mm: float = 45.0
    max_crown_to_chin_mm: float = 36.0
    min_crown_to_chin_mm: float = 31.0
    target_crown_to_chin_mm: float = 34.0
    max_extra_padding_px: int = 600


class FaceFramingPipeline:
    def __init__(self,
                 img: np.ndarray,
                 model: YOLO,
                 params: RunParameters,
                 logdir: str,
                 save_debug: bool = True) -> None:
        self.img = img
        self.model = model
        self.params = params
        self.logdir = logdir
        self.save_debug = save_debug
        self.items: List[FacePipelineItem] = []

    def run(self) -> List[FacePipelineItem]:
        self._detect_faces()
        if not self.items:
            return []
        self._detect_hair_and_top_margin()
        self._extend_bottom_margin()
        self._crop_and_map()
        self._scale_crops_with_constraints()
        if self.save_debug:
            self._save_debug_outputs()
        return self.items

    # Stage 1 -----------------------------------------------------------------
    def _detect_faces(self) -> None:
        print("[Stage 1] Running YOLO face detection and ratio adjustment…")
        result = self.model.predict(self.img, verbose=False)[0]
        boxes_xyxy = (
            result.boxes.xyxy.detach().cpu().numpy()
            if result.boxes is not None
            else np.empty((0, 4))
        )

        for (x1, y1, x2, y2) in boxes_xyxy:
            ax1, ay1, ax2, ay2 = adjust_box_to_ratio(
                x1,
                y1,
                x2,
                y2,
                img_w=self.img.shape[1],
                img_h=self.img.shape[0],
                target_w_over_h=self.params.target_w_over_h,
                strategy="auto",
            )
            item = FacePipelineItem(
                original_box=(ax1, ay1, ax2, ay2),
                top_margin_box=(ax1, ay1, ax2, ay2),
                bottom_scaled_box=(ax1, ay1, ax2, ay2),
                final_box=(ax1, ay1, ax2, ay2),
                chin_y_abs=ay2,
                top_face_y_abs=ay1,
                original_bottom_y_abs=ay2,
            )
            self.items.append(item)

        print(f"[Stage 1] Found {len(self.items)} face candidate(s).")

    # Stage 2 -----------------------------------------------------------------
    def _detect_hair_and_top_margin(self) -> None:
        print(
            f"[Stage 2] Detecting hair top and applying top margin ratio {self.params.top_margin_ratio:.3f}."
        )
        H, W = self.img.shape[:2]
        for idx, item in enumerate(self.items, start=1):
            y_top, debug = top_of_hair_y_debug(self.img, item.original_box, W, H)
            item.hair_top_y_abs = y_top
            item.debug = debug

            crown_y = None
            if y_top is not None and item.top_face_y_abs is not None:
                crown_y = float(min(y_top, item.top_face_y_abs))
            elif y_top is not None:
                crown_y = float(y_top)
            else:
                crown_y = item.top_face_y_abs
            item.crown_y_abs = crown_y

            if y_top is None:
                print(f"  [Face {idx}] Hair top not detected; retaining original box.")
                item.top_margin_box = item.original_box
                item.top_margin_ratio_achieved = None
                continue

            expanded_box, achieved = expand_box_to_hair_top(
                item.original_box,
                y_top,
                img_w=W,
                img_h=H,
                target_w_over_h=self.params.target_w_over_h,
                margin_ratio=self.params.top_margin_ratio,
            )
            item.top_margin_box = expanded_box
            item.top_margin_ratio_achieved = achieved

            if achieved is None:
                print(f"  [Face {idx}] Top margin expansion applied (ratio not measurable).")
            elif achieved + 1e-3 < self.params.top_margin_ratio:
                print(
                    f"  [Face {idx}] Top margin shortfall: achieved {achieved:.3f} < target {self.params.top_margin_ratio:.3f}."
                )
            else:
                print(f"  [Face {idx}] Top margin satisfied (achieved {achieved:.3f}).")

    # Stage 3 -----------------------------------------------------------------
    def _extend_bottom_margin(self) -> None:
        print(
            f"[Stage 3] Extending box bottom with upper ratio {self.params.bottom_upper_ratio:.3f}."
        )
        H, W = self.img.shape[:2]
        for idx, item in enumerate(self.items, start=1):
            base_box = item.top_margin_box or item.original_box
            new_box, achieved = extend_box_downwards(
                base_box,
                img_w=W,
                img_h=H,
                upper_ratio=self.params.bottom_upper_ratio,
            )
            item.bottom_scaled_box = new_box
            item.final_box = new_box
            item.upper_ratio_achieved = achieved

            if achieved is None:
                print(f"  [Face {idx}] Bottom extension applied (ratio not measurable).")
            elif achieved - 1e-3 > self.params.bottom_upper_ratio:
                print(
                    f"  [Face {idx}] Bottom extension limited by image bounds (achieved {achieved:.3f})."
                )
            else:
                print(f"  [Face {idx}] Bottom extension satisfied (achieved {achieved:.3f}).")

    # Stage 4 -----------------------------------------------------------------
    def _crop_and_map(self) -> None:
        print("[Stage 4] Cropping bottom-scaled boxes and mapping key landmarks.")
        H, W = self.img.shape[:2]
        for idx, item in enumerate(self.items, start=1):
            if item.bottom_scaled_box is None:
                print(f"  [Face {idx}] Missing bottom-scaled box; skipping.")
                continue

            x1, y1, x2, y2 = item.bottom_scaled_box
            xi1 = max(0, min(W - 1, int(math.floor(x1))))
            yi1 = max(0, min(H - 1, int(math.floor(y1))))
            xi2 = max(xi1 + 1, min(W, int(math.ceil(x2))))
            yi2 = max(yi1 + 1, min(H, int(math.ceil(y2))))

            crop = self.img[yi1:yi2, xi1:xi2]
            if crop.size == 0:
                print(f"  [Face {idx}] Crop empty after clamping; skipping.")
                continue

            item.final_box = (x1, y1, x2, y2)
            item.final_crop = crop.copy()

            def local_offset(val: Optional[float]) -> Optional[float]:
                if val is None:
                    return None
                return float(val - yi1)

            item.chin_y_local = local_offset(item.chin_y_abs)
            item.hair_top_y_local = local_offset(item.hair_top_y_abs)
            item.top_face_y_local = local_offset(item.top_face_y_abs)
            item.crown_y_local = local_offset(item.crown_y_abs)
            item.original_bottom_y_local = local_offset(item.original_bottom_y_abs)

            msg_bits = [f"crop {crop.shape[1]}x{crop.shape[0]} px"]
            if item.chin_y_local is not None and item.hair_top_y_local is not None:
                span = max(0.0, item.chin_y_local - item.hair_top_y_local)
                msg_bits.append(f"hair-chin {span:.1f} px")
            if item.chin_y_local is not None and item.top_face_y_local is not None:
                span = max(0.0, item.chin_y_local - item.top_face_y_local)
                msg_bits.append(f"forehead-chin {span:.1f} px")
            if item.chin_y_local is not None and item.crown_y_local is not None:
                span = max(0.0, item.chin_y_local - item.crown_y_local)
                msg_bits.append(f"crown-chin {span:.1f} px")
            print(f"  [Face {idx}] {' | '.join(msg_bits)}")

    # Stage 5 -----------------------------------------------------------------
    def _scale_crops_with_constraints(self) -> None:
        print(
            "[Stage 5] Scaling crops to meet pixel/mm constraints and trimming padding."
        )
        if self.save_debug:
            os.makedirs(os.path.join(self.logdir, "final_crops"), exist_ok=True)

        for idx, item in enumerate(self.items, start=1):
            crop = item.final_crop
            if crop is None or crop.size == 0:
                print(f"  [Face {idx}] No crop available; skipping.")
                continue

            crop_h, crop_w = crop.shape[:2]
            chin = item.chin_y_local
            hair = item.hair_top_y_local
            forehead = item.top_face_y_local
            crown = item.crown_y_local
            bottom = item.original_bottom_y_local

            crown_span_px = None
            if chin is not None and crown is not None:
                crown_span_px = max(0.0, chin - crown)

            min_height_req_px: Optional[int] = None
            max_height_req_px: Optional[int] = None
            pad_limited = False
            trim_limited = False

            if (
                crown_span_px is not None
                and crown_span_px > 0
                and self.params.max_crown_to_chin_mm > 0
                and self.params.min_crown_to_chin_mm > 0
            ):
                min_height_bound = (
                    crown_span_px * self.params.target_height_mm
                ) / self.params.max_crown_to_chin_mm
                max_height_bound = (
                    crown_span_px * self.params.target_height_mm
                ) / self.params.min_crown_to_chin_mm

                min_height_req_px = int(math.ceil(min_height_bound))
                max_height_req_px = int(math.floor(max_height_bound))
                if max_height_req_px < min_height_req_px:
                    max_height_req_px = min_height_req_px

            desired_height = float(crop_h)
            if (
                crown_span_px is not None
                and crown_span_px > 0
                and self.params.target_crown_to_chin_mm > 0
            ):
                desired_height = (
                    crown_span_px * self.params.target_height_mm
                ) / self.params.target_crown_to_chin_mm

            if min_height_req_px is not None:
                desired_height = max(desired_height, float(min_height_req_px))
            if max_height_req_px is not None:
                desired_height = min(desired_height, float(max_height_req_px))

            min_possible_height = 1.0
            if crown_span_px is not None:
                min_possible_height = max(
                    min_possible_height,
                    float(int(math.ceil(crown_span_px)) + 1),
                )
            desired_height = max(desired_height, min_possible_height)

            max_possible_height = float(crop_h + self.params.max_extra_padding_px)
            if max_height_req_px is not None:
                max_possible_height = min(max_possible_height, float(max_height_req_px))

            if desired_height > max_possible_height + 1e-3:
                pad_limited = True
                desired_height = max_possible_height

            desired_height_int = max(1, int(round(desired_height)))

            if desired_height_int < min_possible_height:
                trim_limited = True
                desired_height_int = int(math.ceil(min_possible_height))

            if min_height_req_px is not None and desired_height_int < min_height_req_px:
                desired_height_int = min_height_req_px
            if max_height_req_px is not None and desired_height_int > max_height_req_px:
                pad_limited = True
                desired_height_int = max_height_req_px

            if desired_height_int > crop_h:
                pad_total = desired_height_int - crop_h
                max_allowed_height = crop_h + self.params.max_extra_padding_px
                if desired_height_int > max_allowed_height:
                    pad_limited = True
                    desired_height_int = max_allowed_height
                    pad_total = max(0, desired_height_int - crop_h)
                pad_top = pad_total // 2
                pad_bottom = pad_total - pad_top
                if pad_top > 0 or pad_bottom > 0:
                    crop = cv2.copyMakeBorder(
                        crop,
                        pad_top,
                        pad_bottom,
                        0,
                        0,
                        cv2.BORDER_CONSTANT,
                        value=(255, 255, 255),
                    )
                    crop_h, crop_w = crop.shape[:2]
                    if chin is not None:
                        chin += pad_top
                    if hair is not None:
                        hair += pad_top
                    if forehead is not None:
                        forehead += pad_top
                    if crown is not None:
                        crown += pad_top
                    if bottom is not None:
                        bottom += pad_top

            elif desired_height_int < crop_h:
                trim_total = crop_h - desired_height_int
                available_top = int(max(0.0, math.floor(crown) if crown is not None else 0.0))
                available_bottom = int(
                    max(
                        0.0,
                        math.floor(crop_h - 1 - chin) if chin is not None else 0.0,
                    )
                )
                actual_trim = min(trim_total, available_top + available_bottom)
                if actual_trim <= 0:
                    trim_limited = True
                else:
                    if actual_trim < trim_total:
                        trim_limited = True
                    trim_top = min(available_top, actual_trim // 2)
                    trim_bottom = actual_trim - trim_top
                    if trim_bottom > available_bottom:
                        excess = trim_bottom - available_bottom
                        trim_bottom = available_bottom
                        trim_top = min(available_top, trim_top + excess)
                    if trim_top > available_top:
                        trim_top = available_top

                    if trim_top + trim_bottom > 0:
                        crop = crop[trim_top:crop_h - trim_bottom, :]
                        crop_h, crop_w = crop.shape[:2]

                        def adjust_after_trim(val: Optional[float]) -> Optional[float]:
                            if val is None:
                                return None
                            return float(np.clip(val - trim_top, 0.0, crop_h - 1.0))

                        chin = adjust_after_trim(chin)
                        hair = adjust_after_trim(hair)
                        forehead = adjust_after_trim(forehead)
                        crown = adjust_after_trim(crown)
                        bottom = adjust_after_trim(bottom)
                    else:
                        trim_limited = True

            scale = max(
                1.0,
                self.params.min_height_px / max(1, crop_h),
                self.params.min_width_px / max(1, crop_w),
            )
            scaled_w = max(1, int(round(crop_w * scale)))
            scaled_h = max(1, int(round(crop_h * scale)))
            final_crop = cv2.resize(crop, (scaled_w, scaled_h), interpolation=cv2.INTER_CUBIC)

            offsets = {
                "chin": chin * scale if chin is not None else None,
                "hair": hair * scale if hair is not None else None,
                "forehead": forehead * scale if forehead is not None else None,
                "crown": crown * scale if crown is not None else None,
                "bottom": bottom * scale if bottom is not None else None,
            }

            px_per_mm = scaled_h / self.params.target_height_mm
            crown_mm = None
            if offsets["chin"] is not None and offsets["crown"] is not None:
                crown_mm = max(0.0, offsets["chin"] - offsets["crown"]) / px_per_mm
                if crown_mm < self.params.min_crown_to_chin_mm - 1e-2:
                    print(
                        f"  [Face {idx}] WARNING crown-to-chin {crown_mm:.2f} mm below "
                        f"{self.params.min_crown_to_chin_mm:.1f} mm."
                    )
                if crown_mm > self.params.max_crown_to_chin_mm + 1e-2:
                    print(
                        f"  [Face {idx}] WARNING crown-to-chin {crown_mm:.2f} mm exceeds "
                        f"{self.params.max_crown_to_chin_mm:.1f} mm."
                    )

            hair_mm = None
            forehead_mm = None
            if offsets["chin"] is not None and offsets["hair"] is not None:
                hair_mm = max(0.0, offsets["chin"] - offsets["hair"]) / px_per_mm
            if offsets["chin"] is not None and offsets["forehead"] is not None:
                forehead_mm = max(0.0, offsets["chin"] - offsets["forehead"]) / px_per_mm

            if pad_limited and min_height_req_px is not None and crop_h < min_height_req_px:
                print(
                    f"  [Face {idx}] NOTE padding limited by {self.params.max_extra_padding_px}px cap; "
                    "crown-to-chin minimum unmet."
                )
            if trim_limited and max_height_req_px is not None and crop_h > max_height_req_px:
                print(
                    f"  [Face {idx}] NOTE insufficient room to trim top/bottom without cutting features; "
                    "crown-to-chin maximum unmet."
                )

            item.final_crop = final_crop
            item.chin_y_local = offsets["chin"]
            item.hair_top_y_local = offsets["hair"]
            item.top_face_y_local = offsets["forehead"]
            item.crown_y_local = offsets["crown"]
            item.original_bottom_y_local = offsets["bottom"]

            if self.save_debug:
                crops_dir = os.path.join(self.logdir, "final_crops")
                raw_path = os.path.join(crops_dir, f"face{idx:02d}_crop_original.jpg")
                cv2.imwrite(raw_path, crop)

                crop_path = os.path.join(crops_dir, f"face{idx:02d}_crop.jpg")
                cv2.imwrite(crop_path, final_crop)

                annotated_crop = final_crop.copy()
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                label_thickness = 1

                def annotate_line(name: str,
                                  offset: Optional[float],
                                  color: Tuple[int, int, int]) -> None:
                    if offset is None:
                        return
                    y = int(np.clip(round(offset), 0, annotated_crop.shape[0] - 1))
                    cv2.line(annotated_crop, (0, y), (annotated_crop.shape[1] - 1, y), color, 2)
                    text_y = int(np.clip(y - 6, 12, annotated_crop.shape[0] - 6))
                    cv2.putText(
                        annotated_crop,
                        name,
                        (8, text_y + 1),
                        font,
                        font_scale,
                        (0, 0, 0),
                        thickness=label_thickness + 1,
                        lineType=cv2.LINE_AA,
                    )
                    cv2.putText(
                        annotated_crop,
                        name,
                        (8, text_y),
                        font,
                        font_scale,
                        color,
                        thickness=label_thickness,
                        lineType=cv2.LINE_AA,
                    )

                line_specs = [
                    ("Crown", item.crown_y_local, (255, 255, 0)),
                    ("Hair Top", item.hair_top_y_local, (0, 0, 255)),
                    ("Forehead", item.top_face_y_local, (0, 255, 255)),
                    ("Chin", item.chin_y_local, (0, 165, 255)),
                    ("Bottom", item.original_bottom_y_local, (255, 0, 255)),
                ]

                for name, offset, color in line_specs:
                    annotate_line(name, offset, color)

                distance_specs: List[Tuple[str, Optional[float], Tuple[int, int, int]]] = [
                    ("Crown→Chin", crown_mm, (255, 255, 0)),
                    ("Hair→Chin", hair_mm, (0, 0, 255)),
                    ("Forehead→Chin", forehead_mm, (0, 255, 255)),
                ]
                legend_entries = [
                    (label, value, color)
                    for (label, value, color) in distance_specs
                    if value is not None
                ]
                if legend_entries:
                    max_text_width = 0
                    for label, value, color in legend_entries:
                        text = f"{label}: {value:.2f} mm"
                        (text_w, _), _ = cv2.getTextSize(
                            text,
                            font,
                            font_scale,
                            label_thickness,
                        )
                        max_text_width = max(max_text_width, text_w)
                    legend_x = max(8, annotated_crop.shape[1] - max_text_width - 12)
                    for idx_entry, (label, value, color) in enumerate(legend_entries):
                        text = f"{label}: {value:.2f} mm"
                        text_y = 20 + idx_entry * 20
                        cv2.putText(
                            annotated_crop,
                            text,
                            (legend_x, text_y + 1),
                            font,
                            font_scale,
                            (0, 0, 0),
                            thickness=label_thickness + 1,
                            lineType=cv2.LINE_AA,
                        )
                        cv2.putText(
                            annotated_crop,
                            text,
                            (legend_x, text_y),
                            font,
                            font_scale,
                            color,
                            thickness=label_thickness,
                            lineType=cv2.LINE_AA,
                        )

                annotated_path = os.path.join(crops_dir, f"face{idx:02d}_crop_guides.jpg")
                cv2.imwrite(annotated_path, annotated_crop)

                metrics = []
                if crown_mm is not None:
                    metrics.append(f"crown-chin {crown_mm:.2f} mm")
                if hair_mm is not None:
                    metrics.append(f"hair-chin {hair_mm:.2f} mm")
                if forehead_mm is not None:
                    metrics.append(f"forehead-chin {forehead_mm:.2f} mm")

                abs_coords = []
                if item.chin_y_abs is not None:
                    abs_coords.append(f"chin y {item.chin_y_abs:.1f}")
                if item.top_face_y_abs is not None:
                    abs_coords.append(f"forehead y {item.top_face_y_abs:.1f}")
                if item.hair_top_y_abs is not None:
                    abs_coords.append(f"top hair y {item.hair_top_y_abs:.1f}")
                if item.crown_y_abs is not None:
                    abs_coords.append(f"crown y {item.crown_y_abs:.1f}")
                if item.original_bottom_y_abs is not None:
                    abs_coords.append(f"orig bottom y {item.original_bottom_y_abs:.1f}")

                suffix = ""
                details = metrics + abs_coords
                if details:
                    suffix = " | " + " | ".join(details)

                print(
                    f"  [Face {idx}] Saved crop {scaled_w}x{scaled_h}px -> {crop_path}; "
                    f"guides -> {annotated_path}{suffix}"
                )
            else:
                summary = [f"size {final_crop.shape[1]}x{final_crop.shape[0]} px"]
                if crown_mm is not None:
                    summary.append(f"crown-chin {crown_mm:.2f} mm")
                if hair_mm is not None:
                    summary.append(f"hair-chin {hair_mm:.2f} mm")
                if forehead_mm is not None:
                    summary.append(f"forehead-chin {forehead_mm:.2f} mm")
                print(f"  [Face {idx}] Final crop in memory ({' | '.join(summary)})")

    # Debug saving ------------------------------------------------------------
    def _save_debug_outputs(self) -> None:
        detection_vis = self._create_detection_visual()
        hair_vis = self._create_hair_visual()
        final_vis = self._create_final_visual()

        detection_path = os.path.join(self.logdir, "faces_ratio_adjusted.jpg")
        final_path = os.path.join(self.logdir, "faces_7x9_final_boxes.jpg")
        top_margin_path = os.path.join(self.logdir, "faces_7x9_top_margin.jpg")
        hair_dir = os.path.join(self.logdir, "hair_debug")
        os.makedirs(hair_dir, exist_ok=True)

        cv2.imwrite(detection_path, detection_vis)
        cv2.imwrite(final_path, final_vis)
        cv2.imwrite(top_margin_path, final_vis)
        cv2.imwrite(os.path.join(hair_dir, "result_with_top_lines.jpg"), hair_vis)

        self._save_hair_debug_steps(hair_dir)

        print(f"[Output] Ratio-adjusted detections -> {detection_path}")
        print(f"[Output] Final boxes visualization -> {final_path}")
        print(f"[Output] Hair-line diagnostics -> {os.path.join(hair_dir, 'result_with_top_lines.jpg')}")

    def _create_detection_visual(self) -> np.ndarray:
        vis = self.img.copy()
        for item in self.items:
            x1, y1, x2, y2 = item.original_box
            cv2.rectangle(
                vis,
                (int(round(x1)), int(round(y1))),
                (int(round(x2)), int(round(y2))),
                (0, 255, 0),
                3,
            )
        return vis

    def _create_hair_visual(self) -> np.ndarray:
        vis = self.img.copy()
        H, W = self.img.shape[:2]
        for item in self.items:
            x1, y1, x2, y2 = item.original_box
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
            if item.hair_top_y_abs is not None:
                cx = int(round((x1 + x2) / 2.0))
                cy = int(np.clip(round(item.hair_top_y_abs), 0, H - 1))
                cv2.circle(vis, (cx, cy), 6, (0, 0, 255), -1)
                cv2.line(vis, (0, cy), (W - 1, cy), (0, 0, 255), 2)
        return vis

    def _create_final_visual(self) -> np.ndarray:
        vis = self.img.copy()
        H = vis.shape[0]
        for item in self.items:
            if item.final_box is None:
                continue
            x1, y1, x2, y2 = item.final_box
            cv2.rectangle(
                vis,
                (int(round(x1)), int(round(y1))),
                (int(round(x2)), int(round(y2))),
                (255, 0, 0),
                3,
            )
            if item.chin_y_abs is not None:
                cy = int(np.clip(round(item.chin_y_abs), 0, H - 1))
                cv2.line(vis, (int(round(x1)), cy), (int(round(x2)), cy), (0, 165, 255), 2)
            if item.hair_top_y_abs is not None:
                hy = int(np.clip(round(item.hair_top_y_abs), 0, H - 1))
                cv2.line(vis, (int(round(x1)), hy), (int(round(x2)), hy), (0, 0, 255), 2)
            if item.top_face_y_abs is not None:
                ty = int(np.clip(round(item.top_face_y_abs), 0, H - 1))
                cv2.line(vis, (int(round(x1)), ty), (int(round(x2)), ty), (0, 255, 255), 2)
        return vis

    def _save_hair_debug_steps(self, out_dir: str) -> None:
        steps = [
            "roi_bgr",
            "seed_mask_color",
            "gc_mask_color",
            "fg_only_overlay",
            "main_comp_overlay",
            "band_overlay",
        ]
        for idx, item in enumerate(self.items, start=1):
            for step in steps:
                if step in item.debug:
                    path = os.path.join(out_dir, f"face{idx:02d}_{step}.png")
                    cv2.imwrite(path, item.debug[step])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hair-aware portrait framing pipeline")
    parser.add_argument("--img-path", type=str, default="./test_image.jpg", help="Path to the input image")
    parser.add_argument("--logdir", type=str, default="./logs", help="Directory where outputs/debug images are saved")
    parser.add_argument("--top-margin-ratio", dest="top_margin_ratio", type=float, default=0.10,
                        help="Fraction of the final box height kept above the hair line")
    parser.add_argument("--bottom-upper-ratio", dest="bottom_upper_ratio", type=float, default=0.80,
                        help="Fraction of the final box height occupied before bottom extension")
    parser.add_argument("--w-over-h", dest="w_over_h", type=float, default=7.0 / 9.0,
                        help="Target width/height ratio for adjusted boxes")
    parser.add_argument("--min-width-px", dest="min_width_px", type=int, default=420,
                        help="Minimum width of the final crop in pixels")
    parser.add_argument("--min-height-px", dest="min_height_px", type=int, default=540,
                        help="Minimum height of the final crop in pixels")
    parser.add_argument("--crown-chin-max-mm", "--hair-chin-max-mm",
                        dest="crown_chin_max_mm", type=float, default=36.0,
                        help="Maximum allowed crown-to-chin distance in millimetres")
    parser.add_argument("--crown-chin-min-mm", "--forehead-chin-min-mm",
                        dest="crown_chin_min_mm", type=float, default=31.0,
                        help="Minimum required crown-to-chin distance in millimetres")
    parser.add_argument("--crown-chin-target-mm", dest="crown_chin_target_mm", type=float, default=35.0,
                        help="Target crown-to-chin distance in millimetres")
    parser.add_argument("--target-height-mm", dest="target_height_mm", type=float, default=45.0,
                        help="Reference portrait height in millimetres")
    parser.add_argument("--no-debug", dest="no_debug", action="store_true",
                        help="Disable saving debug images to disk")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    params = RunParameters(
        target_w_over_h=args.w_over_h,
        top_margin_ratio=args.top_margin_ratio,
        bottom_upper_ratio=args.bottom_upper_ratio,
        min_width_px=args.min_width_px,
        min_height_px=args.min_height_px,
        target_height_mm=args.target_height_mm,
        max_crown_to_chin_mm=args.crown_chin_max_mm,
        min_crown_to_chin_mm=args.crown_chin_min_mm,
        target_crown_to_chin_mm=args.crown_chin_target_mm,
    )

    save_debug = not args.no_debug
    os.makedirs(args.logdir, exist_ok=True)

    print("=== Hair-Aware Passport Crop Pipeline ===")
    print(f"Input image: {args.img_path}")
    print(
        f"Params | ratio {params.target_w_over_h:.3f} | top margin {params.top_margin_ratio:.3f} | "
        f"bottom upper {params.bottom_upper_ratio:.3f} | min size {params.min_width_px}x{params.min_height_px}"
    )
    if not save_debug:
        print("[Info] Debug image saving disabled; results kept in memory only.")

    img = cv2.imread(args.img_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {args.img_path}")

    model = YOLO("./yolo-face.pt")

    pipeline = FaceFramingPipeline(
        img=img,
        model=model,
        params=params,
        logdir=args.logdir,
        save_debug=save_debug,
    )

    items = pipeline.run()
    if not items:
        print("[Done] No faces to process.")
        return

    print(f"[Done] Processed {len(items)} face(s).")
    if save_debug:
        print(f"[Done] Final crops saved under {os.path.join(args.logdir, 'final_crops')}")


if __name__ == "__main__":
    main(parse_args())
