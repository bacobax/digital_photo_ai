"""High-level interface for the face framing pipeline.

This module exposes a reusable API that can be imported by local scripts
or server endpoints alike.  The main entry-point is the ``PortraitFramer``
class which wraps the lower-level pipeline implemented in ``main.py`` and
returns structured results for each detected face, including final crops,
annotated visualisations, and landmark measurements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import math
import numpy as np
import os
from ultralytics import YOLO
from PIL import Image

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
    pre_scale_crop: Optional[np.ndarray] = None
    crown_to_chin_mm: Optional[float] = None
    hair_to_chin_mm: Optional[float] = None
    forehead_to_chin_mm: Optional[float] = None
    px_per_mm: Optional[float] = None
    pad_limited: bool = False
    trim_limited: bool = False
    min_height_req_px: Optional[int] = None
    max_height_req_px: Optional[int] = None
    alpha: Optional[float] = None
    chin_crown_T_low: Optional[float] = None
    chin_crown_T_high: Optional[float] = None

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
    resize_scaling: float = 0.0


class FaceFramingPipeline:
    def _save_with_dpi(self, bgr: np.ndarray, path: str) -> None:
        """Save an image embedding DPI so editors show 35×45 mm correctly.

        DPI is computed dynamically from the current pixel size and the target
        physical size (width_mm = target_w_over_h * target_height_mm,
        height_mm = target_height_mm).
        """
        if bgr is None or bgr.size == 0:
            return
        h_px, w_px = bgr.shape[:2]
        # target physical dimensions in mm
        height_mm = float(self.params.target_height_mm)
        width_mm = float(self.params.target_w_over_h * self.params.target_height_mm)
        # Compute dpi; guard against division by zero
        dpi_x = (w_px / max(1e-6, width_mm)) * 25.4
        dpi_y = (h_px / max(1e-6, height_mm)) * 25.4
        # Convert to RGB and save via PIL so DPI is stored in metadata
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        # For JPEG, PIL writes JFIF density with the provided dpi tuple
        img.save(path, dpi=(dpi_x, dpi_y), quality=95, subsampling=0)
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
        self._adjust_crops_with_trim_padding()
        self._resize_crops_and_map_mm()
        self._balance_lighting()
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

        if boxes_xyxy.shape[0] == 0:
            print("[Stage 1] No faces detected.")
            return

        x1, y1, x2, y2 = boxes_xyxy[0]
        if boxes_xyxy.shape[0] > 1:
            print(
                f"[Stage 1] Multiple faces detected; selecting the first of {boxes_xyxy.shape[0]} candidates."
            )

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

        print("[Stage 1] Using 1 face candidate.")

        if self.save_debug:
            self._save_stage1_debug()

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

        if self.save_debug:
            self._save_stage2_debug()

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

        if self.save_debug:
            self._save_stage3_debug()

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
    def _adjust_crops_with_trim_padding(self) -> None:
        print("[Stage 5] Adjusting crops with trim/padding before resizing…")
        for idx, item in enumerate(self.items, start=1):
            crop = item.final_crop
            if crop is None or crop.size == 0:
                print(f"  [Face {idx}] No crop available; skipping adjustments.")
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

            item.pre_scale_crop = crop.copy()
            item.chin_y_local = chin
            item.hair_top_y_local = hair
            item.top_face_y_local = forehead
            item.crown_y_local = crown
            item.original_bottom_y_local = bottom
            item.pad_limited = pad_limited
            item.trim_limited = trim_limited
            item.min_height_req_px = min_height_req_px
            item.max_height_req_px = max_height_req_px
            item.final_crop = None  # will be produced in Stage 6
            item.px_per_mm = None
            item.crown_to_chin_mm = None
            item.hair_to_chin_mm = None
            item.forehead_to_chin_mm = None

        if self.save_debug:
            self._save_stage5_debug()

    # Stage 6 -----------------------------------------------------------------
    def _resize_crops_and_map_mm(self) -> None:
        print("[Stage 6] Resizing to target size (>= minimum) and mapping chin→crown span…")
        for idx, item in enumerate(self.items, start=1):
            crop = item.pre_scale_crop
            if crop is None or crop.size == 0:
                print(f"  [Face {idx}] No adjusted crop available; skipping.")
                continue

            crop_h, crop_w = crop.shape[:2]
            chin = item.chin_y_local
            crown = item.crown_y_local
            hair = item.hair_top_y_local
            forehead = item.top_face_y_local
            bottom = item.original_bottom_y_local

            resize_scaling = float(getattr(self.params, "resize_scaling", 0.0))
            if not math.isfinite(resize_scaling):
                resize_scaling = 0.0
            resize_scaling = max(0.0, min(1.0, resize_scaling))

            scaled_height = int(round(crop_h * resize_scaling))
            target_height_px = max(self.params.min_height_px, scaled_height)
            target_height_px = max(1, target_height_px)

            scale = target_height_px / max(1, crop_h)
            target_width_px = max(1, int(round(crop_w * scale)))
            final_crop = cv2.resize(
                crop,
                (target_width_px, target_height_px),
                interpolation=cv2.INTER_CUBIC,
            )

            pad_left = pad_right = 0
            if final_crop.shape[1] < self.params.min_width_px:
                pad_total = self.params.min_width_px - final_crop.shape[1]
                pad_left = pad_total // 2
                pad_right = pad_total - pad_left
                final_crop = cv2.copyMakeBorder(
                    final_crop,
                    0,
                    0,
                    pad_left,
                    pad_right,
                    cv2.BORDER_CONSTANT,
                    value=(255, 255, 255),
                )

            scaled_h = final_crop.shape[0]
            scale_px = scaled_h / max(1, crop_h)

            if chin is not None:
                chin *= scale_px
            if crown is not None:
                crown *= scale_px
            if hair is not None:
                hair *= scale_px
            if forehead is not None:
                forehead *= scale_px
            if bottom is not None:
                bottom *= scale_px

            item.final_crop = final_crop
            item.chin_y_local = chin
            item.crown_y_local = crown
            item.hair_top_y_local = hair
            item.top_face_y_local = forehead
            item.original_bottom_y_local = bottom

            alpha = T_low = T_high = T_eff = chin_to_crown_mm = None

            if crown is None or chin is None or chin <= crown:
                print(f"  [Face {idx}] Crown not detected; skipping mm mapping, keeping pixel-only compliance.")
                if self.save_debug:
                    self._save_final_crops(idx, final_crop, chin, crown, alpha, T_low, T_high, T_eff, chin_to_crown_mm)
                continue

            span_px = chin - crown
            alpha = max(1e-6, span_px / scaled_h)
            T_low = self.params.min_crown_to_chin_mm / alpha
            T_high = self.params.max_crown_to_chin_mm / alpha

            interval_empty = T_low > T_high
            if interval_empty:
                if abs(self.params.target_height_mm - T_low) <= abs(self.params.target_height_mm - T_high):
                    T_eff = T_low
                else:
                    T_eff = T_high
            else:
                T_eff = max(T_low, min(self.params.target_height_mm, T_high))

            px_per_mm = scaled_h / T_eff
            chin_to_crown_mm = span_px / px_per_mm
            item.effective_target_height_mm = T_eff
            item.crown_to_chin_mm = chin_to_crown_mm
            item.px_per_mm = px_per_mm
            item.alpha = alpha
            item.chin_crown_T_low = T_low
            item.chin_crown_T_high = T_high
            if forehead is not None and chin is not None:
                item.forehead_to_chin_mm = max(0.0, (chin - forehead) / px_per_mm)
            if hair is not None and chin is not None:
                item.hair_to_chin_mm = max(0.0, (chin - hair) / px_per_mm)

            if item.pad_limited and item.min_height_req_px is not None and crop_h < item.min_height_req_px:
                print(
                    f"  [Face {idx}] NOTE padding limited by {self.params.max_extra_padding_px}px cap; "
                    "crown-to-chin minimum unmet."
                )
            if item.trim_limited and item.max_height_req_px is not None and crop_h > item.max_height_req_px:
                print(
                    f"  [Face {idx}] NOTE insufficient headroom to trim; crown-to-chin maximum unmet."
                )

            msg = (
                f"effective crop T interval (mm): [{T_low:.2f}, {T_high:.2f}] | "
                f"chosen T_eff={T_eff:.2f} | alpha={alpha:.4f} | chin-crown={chin_to_crown_mm:.2f} mm"
            )
            if interval_empty:
                msg += " | interval_empty=True"
            print(f"  [Face {idx}] {msg}")

            if self.save_debug:
                self._save_final_crops(idx, final_crop, chin, crown, alpha, T_low, T_high, T_eff, chin_to_crown_mm)

    # Stage 7 -----------------------------------------------------------------
    def _balance_lighting(self) -> None:
        print("[Stage 7] Harmonising facial lighting across partitions.")
        for idx, item in enumerate(self.items, start=1):
            crop = item.final_crop
            if crop is None or crop.size == 0:
                print(f"  [Face {idx}] No final crop available; skipping lighting balance.")
                continue

            balanced_crop, stats, scale_map_vis = self._apply_partition_lighting_balance(crop)
            item.final_crop = balanced_crop
            if scale_map_vis is not None and self.save_debug:
                self._save_stage7_lighting_debug(idx, scale_map_vis)

            if stats["needs_adjustment"]:
                print(
                    f"  [Face {idx}] Lighting balanced (gain up to {stats['max_gain']:.2f}x," \
                    f" partitions matched to brightest)."
                )
            else:
                print(f"  [Face {idx}] Lighting already uniform; no adjustment applied.")

            self._emit_final_outputs(idx, item)

    def _save_final_crops(self,
                          idx: int,
                          final_crop: np.ndarray,
                          chin: Optional[float],
                          crown: Optional[float],
                          alpha: Optional[float],
                          T_low: Optional[float],
                          T_high: Optional[float],
                          T_eff: Optional[float],
                          chin_mm: Optional[float]) -> None:
        crops_dir = os.path.join(self.logdir, "final_crops")
        os.makedirs(crops_dir, exist_ok=True)

        crop_path = os.path.join(crops_dir, f"face{idx:02d}_crop.jpg")
        self._save_with_dpi(final_crop, crop_path)

        guides = final_crop.copy()
        height, width = guides.shape[:2]

        def draw_line(y: Optional[float], color: Tuple[int, int, int]) -> None:
            if y is None:
                return
            pos = int(np.clip(round(y), 0, height - 1))
            cv2.line(guides, (0, pos), (width - 1, pos), color, 2)

        draw_line(crown, (0, 0, 255))
        draw_line(chin, (0, 165, 255))

        info_lines: List[str] = []
        if alpha is not None and T_low is not None and T_high is not None and T_eff is not None:
            info_lines.append(f"alpha={alpha:.4f} T=[{T_low:.2f},{T_high:.2f}] T*={T_eff:.2f}")
        if chin_mm is not None:
            info_lines.append(f"chin-crown={chin_mm:.2f} mm")
        if crown is None or chin is None:
            info_lines.append("crown/chin missing")

        y_text = 20
        for text in info_lines:
            cv2.putText(
                guides,
                text,
                (10, y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            y_text += 18

        guides_path = os.path.join(crops_dir, f"face{idx:02d}_crop_guides.jpg")
        self._save_with_dpi(guides, guides_path)

    def _apply_partition_lighting_balance(
        self, crop: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, float], Optional[np.ndarray]]:
        h, w = crop.shape[:2]
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        L = lab[:, :, 0].astype(np.float32)

        mid_y = h // 2
        mid_x = w // 2

        masks = {
            "up": np.zeros((h, w), dtype=bool),
            "down": np.zeros((h, w), dtype=bool),
            "left": np.zeros((h, w), dtype=bool),
            "right": np.zeros((h, w), dtype=bool),
        }

        masks["up"][:mid_y, :] = True
        masks["down"][mid_y:, :] = True
        masks["left"][:, :mid_x] = True
        masks["right"][:, mid_x:] = True

        means: Dict[str, float] = {}
        for name, mask in masks.items():
            if mask.any():
                means[name] = float(L[mask].mean())
            else:
                means[name] = 0.0

        target_mean = max(means.values()) if means else 0.0
        if target_mean <= 1e-3:
            stats = {
                "needs_adjustment": False,
                "max_gain": 1.0,
                "target_mean": target_mean,
                "means": means,
            }
            return crop, stats, None

        max_gain_allowed = 1.6
        needs_adjustment = False
        scale_map = np.ones((h, w), dtype=np.float32)
        max_gain_applied = 1.0

        for name, mask in masks.items():
            mean_val = means[name]
            if mean_val <= 1e-3:
                continue
            gain = min(max_gain_allowed, max(1.0, target_mean / (mean_val + 1e-3)))
            if gain > 1.02:
                needs_adjustment = True
            max_gain_applied = max(max_gain_applied, gain)
            if gain > 1.0:
                scale_map[mask] = np.maximum(scale_map[mask], gain)

        if not needs_adjustment:
            stats = {
                "needs_adjustment": False,
                "max_gain": max_gain_applied,
                "target_mean": target_mean,
                "means": means,
            }
            return crop, stats, None

        sigma = max(1.0, min(h, w) / 30.0)
        scale_map = cv2.GaussianBlur(scale_map, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
        scale_map = np.clip(scale_map, 1.0, max_gain_allowed)

        L_balanced = np.clip(L * scale_map, 0, 255)
        lab[:, :, 0] = L_balanced.astype(np.uint8)
        balanced_crop = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        scale_map_vis = cv2.normalize(scale_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        scale_map_vis = cv2.applyColorMap(scale_map_vis, cv2.COLORMAP_INFERNO)

        stats = {
            "needs_adjustment": True,
            "max_gain": max_gain_applied,
            "target_mean": target_mean,
            "means": means,
        }
        return balanced_crop, stats, scale_map_vis

    def _emit_final_outputs(self, idx: int, item: FacePipelineItem) -> None:
        crop = item.final_crop
        if crop is None or crop.size == 0:
            print(f"  [Face {idx}] No final crop to export.")
            return

        metrics = []
        if item.crown_to_chin_mm is not None:
            metrics.append(f"crown-chin {item.crown_to_chin_mm:.2f} mm")
        if item.hair_to_chin_mm is not None:
            metrics.append(f"hair-chin {item.hair_to_chin_mm:.2f} mm")
        if item.forehead_to_chin_mm is not None:
            metrics.append(f"forehead-chin {item.forehead_to_chin_mm:.2f} mm")

        if self.save_debug:
            crops_dir = os.path.join(self.logdir, "stage6_final")
            os.makedirs(crops_dir, exist_ok=True)

            if item.pre_scale_crop is not None:
                cv2.imwrite(
                    os.path.join(crops_dir, f"face{idx:02d}_stage6_input_reference.jpg"),
                    item.pre_scale_crop,
                )

            crop_path = os.path.join(crops_dir, f"face{idx:02d}_stage6_final_balanced.jpg")
            self._save_with_dpi(crop, crop_path)

            annotated_crop = self._build_annotated_crop(item)
            annotated_path = os.path.join(crops_dir, f"face{idx:02d}_stage6_final_guides.jpg")
            self._save_with_dpi(annotated_crop, annotated_path)

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
                f"  [Face {idx}] Stage 7 balanced crop {crop.shape[1]}x{crop.shape[0]}px -> {crop_path}; "
                f"guides -> {annotated_path}{suffix}"
            )

            # Also store balanced crop+guides in shared final_crops directory
            self._save_final_crops(
                idx,
                crop,
                item.chin_y_local,
                item.crown_y_local,
                item.alpha,
                item.chin_crown_T_low,
                item.chin_crown_T_high,
                item.effective_target_height_mm,
                item.crown_to_chin_mm,
            )
        else:
            summary = [f"size {crop.shape[1]}x{crop.shape[0]} px"] + metrics
            print(f"  [Face {idx}] Final crop ready ({' | '.join(summary)})")

    def _build_annotated_crop(self, item: FacePipelineItem) -> np.ndarray:
        if item.final_crop is None:
            return np.empty((0, 0, 3), dtype=np.uint8)

        annotated = item.final_crop.copy()
        h, w = annotated.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        label_font_scale = 0.55
        label_thickness = 1

        line_thickness = max(2, h // 180 + 1)

        def annotate_marker(name: str,
                            offset: Optional[float],
                            color: Tuple[int, int, int]) -> None:
            if offset is None:
                return
            y = int(np.clip(round(offset), 0, h - 1))
            cv2.line(annotated, (0, y), (w - 1, y), color, line_thickness)

            text = name
            text_size, baseline = cv2.getTextSize(text, font, label_font_scale, label_thickness)
            text_w, text_h = text_size
            pad = 4

            # Prefer placing the label above the marker; fallback below if cramped
            text_x = 8
            baseline_y = y - 8
            if baseline_y - text_h - baseline < 4:
                baseline_y = min(h - 5, y + text_h + baseline + 6)

            bg_top = max(2, baseline_y - text_h - baseline - pad)
            bg_bottom = min(h - 2, baseline_y + pad)
            bg_left = max(2, text_x - pad)
            bg_right = min(w - 2, text_x + text_w + pad)

            cv2.rectangle(annotated, (bg_left, bg_top), (bg_right, bg_bottom), (0, 0, 0), -1)
            cv2.putText(
                annotated,
                text,
                (text_x, baseline_y),
                font,
                label_font_scale,
                color,
                thickness=label_thickness,
                lineType=cv2.LINE_AA,
            )

        markers = [
            ("Crown", item.crown_y_local, (0, 0, 255)),
            ("Forehead", item.top_face_y_local, (0, 165, 255)),
            ("Chin", item.chin_y_local, (0, 255, 0)),
        ]

        for name, offset, color in markers:
            annotate_marker(name, offset, color)

        arrow_specs = [
            (item.crown_y_local, item.chin_y_local, item.crown_to_chin_mm, (0, 0, 255), "crown-chin"),
            (item.top_face_y_local, item.chin_y_local, item.forehead_to_chin_mm, (0, 165, 255), "forehead-chin"),
        ]

        base_x = w - 35
        step_x = 55
        number_font_scale = 0.7
        number_thickness = 2

        for idx, (start_offset, end_offset, value_mm, color, label) in enumerate(arrow_specs):
            if start_offset is None or end_offset is None:
                continue
            y_start = int(round(start_offset))
            y_end = int(round(end_offset))
            x = max(20, base_x - idx * step_x)
            px_span = abs(y_end - y_start)
            if value_mm is not None:
                measurement = f"{label}: {px_span:.0f}px | {value_mm:.2f} mm"
            else:
                measurement = f"{label}: {px_span:.0f}px"
            self._draw_distance_arrow(
                annotated,
                y_start,
                y_end,
                x,
                color,
                measurement,
                number_font_scale,
                number_thickness,
            )

        return annotated

    @staticmethod
    def _draw_distance_arrow(
        img: np.ndarray,
        y_start: int,
        y_end: int,
        x: int,
        color: Tuple[int, int, int],
        label: str,
        font_scale: float,
        thickness: int,
    ) -> None:
        h, w = img.shape[:2]
        y_top = int(np.clip(min(y_start, y_end), 0, h - 1))
        y_bottom = int(np.clip(max(y_start, y_end), 0, h - 1))
        if y_bottom - y_top < 2:
            return

        x = int(np.clip(x, 5, w - 5))
        cv2.arrowedLine(img, (x, y_top), (x, y_bottom), color, 2, tipLength=0.12)
        cv2.arrowedLine(img, (x, y_bottom), (x, y_top), color, 2, tipLength=0.12)

        mid_y = (y_top + y_bottom) // 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        rect_left = x + 10
        rect_right = rect_left + text_w + 10
        if rect_right >= w - 5:
            rect_left = max(5, w - text_w - 20)
            rect_right = rect_left + text_w + 10

        rect_top = max(5, mid_y - text_h // 2 - 6)
        rect_bottom = min(h - 5, mid_y + text_h // 2 + baseline + 6)

        cv2.rectangle(
            img,
            (rect_left, rect_top),
            (rect_right, rect_bottom),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            img,
            label,
            (rect_left + 5, rect_bottom - baseline - 3),
            font,
            font_scale,
            color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

    # Debug helpers -----------------------------------------------------------
    def _save_stage1_debug(self) -> None:
        stage_dir = os.path.join(self.logdir, "stage1_detection")
        os.makedirs(stage_dir, exist_ok=True)
        path = os.path.join(stage_dir, "stage1_ratio_adjusted_boxes.jpg")
        cv2.imwrite(path, self._create_detection_visual())
        print(f"[Stage 1] Debug saved -> {path}")

    def _save_stage2_debug(self) -> None:
        stage_dir = os.path.join(self.logdir, "stage2_hair")
        steps_dir = os.path.join(stage_dir, "steps")
        os.makedirs(stage_dir, exist_ok=True)
        os.makedirs(steps_dir, exist_ok=True)

        overlay_path = os.path.join(stage_dir, "stage2_hairline_overlay.jpg")
        cv2.imwrite(overlay_path, self._create_hair_visual())

        step_names = [
            "roi_bgr",
            "seed_mask_color",
            "gc_mask_color",
            "fg_only_overlay",
            "main_comp_overlay",
            "band_overlay",
        ]
        for idx, item in enumerate(self.items, start=1):
            for step in step_names:
                if step in item.debug:
                    step_path = os.path.join(steps_dir, f"face{idx:02d}_{step}.png")
                    cv2.imwrite(step_path, item.debug[step])

        print(f"[Stage 2] Debug saved -> {overlay_path} and per-face steps in {steps_dir}")

    def _save_stage3_debug(self) -> None:
        stage_dir = os.path.join(self.logdir, "stage3_boxes")
        os.makedirs(stage_dir, exist_ok=True)
        path = os.path.join(stage_dir, "stage3_bottom_extension_overlay.jpg")
        cv2.imwrite(path, self._create_final_visual())
        print(f"[Stage 3] Debug saved -> {path}")

    def _save_stage5_debug(self) -> None:
        stage_dir = os.path.join(self.logdir, "stage5_adjusted")
        os.makedirs(stage_dir, exist_ok=True)
        for idx, item in enumerate(self.items, start=1):
            if item.pre_scale_crop is not None:
                pre_path = os.path.join(stage_dir, f"face{idx:02d}_adjusted_crop.jpg")
                cv2.imwrite(pre_path, item.pre_scale_crop)
        print(f"[Stage 5] Debug (pre-scale crops) saved under {stage_dir}")

    def _save_stage7_lighting_debug(self, idx: int, scale_map_vis: np.ndarray) -> None:
        stage_dir = os.path.join(self.logdir, "stage7_lighting")
        os.makedirs(stage_dir, exist_ok=True)
        path = os.path.join(stage_dir, f"face{idx:02d}_stage7_lighting_gain.jpg")
        cv2.imwrite(path, scale_map_vis)
        print(f"  [Face {idx}] Lighting gain map saved -> {path}")

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



@dataclass
class LandmarkMeasurement:
    """Capture a landmark's vertical position in pixels and millimetres."""

    px: Optional[float]
    mm: Optional[float]


@dataclass
class FaceProcessingResult:
    """Structured output for a single processed face."""

    final_image: np.ndarray
    annotated_image: np.ndarray
    crown_to_chin_mm: Optional[float]
    hair_to_chin_mm: Optional[float]
    forehead_to_chin_mm: Optional[float]
    chin: LandmarkMeasurement
    crown: LandmarkMeasurement
    forehead: LandmarkMeasurement
    px_per_mm: Optional[float]

    def to_dict(self) -> Dict[str, Optional[float]]:
        """Return a dictionary with scalar measurements only."""

        return {
            "crown_to_chin_mm": self.crown_to_chin_mm,
            "hair_to_chin_mm": self.hair_to_chin_mm,
            "forehead_to_chin_mm": self.forehead_to_chin_mm,
            "chin_px": self.chin.px,
            "chin_mm": self.chin.mm,
            "crown_px": self.crown.px,
            "crown_mm": self.crown.mm,
            "forehead_px": self.forehead.px,
            "forehead_mm": self.forehead.mm,
            "px_per_mm": self.px_per_mm,
        }


class PortraitFramer:
    """Reusable facade for the face framing pipeline."""

    def __init__(self, model_path: str = "./yolo-face.pt") -> None:
        self.model = YOLO(model_path)

    def _ensure_params(self, params: Optional[RunParameters]) -> RunParameters:
        if params is not None:
            return params
        return RunParameters()

    def process_array(
        self,
        image: np.ndarray,
        *,
        logdir: str = "./logs",
        save_debug: bool = False,
        params: Optional[RunParameters] = None,
    ) -> List[FaceProcessingResult]:
        """Run the pipeline on an already-loaded image array."""

        effective_params = self._ensure_params(params)
        pipeline = FaceFramingPipeline(
            img=image,
            model=self.model,
            params=effective_params,
            logdir=logdir,
            save_debug=save_debug,
        )

        items = pipeline.run()
        results: List[FaceProcessingResult] = []

        for item in items:
            if item.final_crop is None:
                continue

            annotated = pipeline._build_annotated_crop(item)
            px_per_mm = item.px_per_mm
            final_height = item.final_crop.shape[0]

            def to_measurement(px_value: Optional[float]) -> LandmarkMeasurement:
                if px_value is None:
                    return LandmarkMeasurement(px=None, mm=None)
                # Convert from top-origin coordinates to distance measured from the bottom edge.
                px_from_bottom = max(0.0, float((final_height - 1) - px_value))
                mm_value = None
                if px_per_mm:
                    mm_value = px_from_bottom / px_per_mm
                return LandmarkMeasurement(px=px_from_bottom, mm=mm_value)

            results.append(
                FaceProcessingResult(
                    final_image=item.final_crop.copy(),
                    annotated_image=annotated,
                    crown_to_chin_mm=item.crown_to_chin_mm,
                    hair_to_chin_mm=item.hair_to_chin_mm,
                    forehead_to_chin_mm=item.forehead_to_chin_mm,
                    chin=to_measurement(item.chin_y_local),
                    crown=to_measurement(item.crown_y_local),
                    forehead=to_measurement(item.top_face_y_local),
                    px_per_mm=px_per_mm,
                )
            )

        return results

    def process_path(
        self,
        img_path: str,
        *,
        logdir: str = "./logs",
        save_debug: bool = False,
        params: Optional[RunParameters] = None,
    ) -> List[FaceProcessingResult]:
        """Read an image from disk and process it."""

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image at {img_path}")
        return self.process_array(
            image,
            logdir=logdir,
            save_debug=save_debug,
            params=params,
        )


def process_portraits(
    img_paths: Iterable[str],
    *,
    logdir: str = "./logs",
    save_debug: bool = False,
    params: Optional[RunParameters] = None,
    model_path: str = "./yolo-face.pt",
) -> Dict[str, List[FaceProcessingResult]]:
    """Convenience helper for batch processing multiple image paths."""

    framer = PortraitFramer(model_path=model_path)
    output: Dict[str, List[FaceProcessingResult]] = {}
    for path in img_paths:
        output[path] = framer.process_path(
            path,
            logdir=logdir,
            save_debug=save_debug,
            params=params,
        )
    return output
