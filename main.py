import argparse
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from sympy import pprint
from pprint import pprint
import cv2
import numpy as np
from ultralytics import YOLO
from portrait_framer import process_portraits, RunParameters


Box = Tuple[float, float, float, float]





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
    parser.add_argument("--crown-chin-target-mm", dest="crown_chin_target_mm", type=float, default=34.0,
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

    output = process_portraits(
        img_paths=[args.img_path],
        logdir=args.logdir,
        params=params,
        save_debug=save_debug,
        model_path="./yolo-face.pt"
    )

    pprint(f"[Done] Processed {len(output)} faces.")
    pprint(output)
    if save_debug:
        print(f"[Done] Final crops saved under {os.path.join(args.logdir, 'final_crops')}")


if __name__ == "__main__":
    main(parse_args())
