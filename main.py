import argparse
import math
import os
from typing import Optional, Tuple

from pprint import pprint

from portrait_framer import process_portraits, RunParameters


Box = Tuple[float, float, float, float]





def _clamp_resize_scaling(value: Optional[float]) -> float:
    """Clamp user-provided scaling to [0, 1]; default to 0 for legacy behaviour."""

    if value is None:
        return 0.0
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(value_f):
        return 0.0
    return max(0.0, min(1.0, value_f))


def parse_args() -> argparse.Namespace:
    """Configure command-line options for the portrait framing demo."""

    parser = argparse.ArgumentParser(
        description="Hair-aware portrait framing pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False,
    )
    parser.add_argument(
        "-h",
        "--help",
        "--h",
        action="help",
        help="Show this help message and exit.",
    )
    parser.add_argument(
        "--img-path",
        type=str,
        default="./test_image.jpg",
        help="Path to the input image",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="./logs",
        help="Directory where outputs/debug images are saved",
    )
    parser.add_argument(
        "--top-margin-ratio",
        dest="top_margin_ratio",
        type=float,
        default=0.10,
        help="Fraction of the final box height kept above the hair line",
    )
    parser.add_argument(
        "--bottom-upper-ratio",
        dest="bottom_upper_ratio",
        type=float,
        default=0.80,
        help="Fraction of the final box height occupied before bottom extension",
    )
    parser.add_argument(
        "--w-over-h",
        dest="w_over_h",
        type=float,
        default=7.0 / 9.0,
        help="Target width/height ratio for adjusted boxes",
    )
    parser.add_argument(
        "--min-width-px",
        dest="min_width_px",
        type=int,
        default=420,
        help="Minimum width of the final crop in pixels",
    )
    parser.add_argument(
        "--min-height-px",
        dest="min_height_px",
        type=int,
        default=540,
        help="Minimum height of the final crop in pixels",
    )
    parser.add_argument(
        "--resize-scaling",
        dest="resize_scaling",
        type=float,
        default=None,
        help="Optional scaling factor (0-1) applied to the pre-resize crop height/width",
    )
    parser.add_argument(
        "--crown-chin-max-mm",
        "--hair-chin-max-mm",
        dest="crown_chin_max_mm",
        type=float,
        default=36.0,
        help="Maximum allowed crown-to-chin distance in millimetres",
    )
    parser.add_argument(
        "--crown-chin-min-mm",
        "--forehead-chin-min-mm",
        dest="crown_chin_min_mm",
        type=float,
        default=31.0,
        help="Minimum required crown-to-chin distance in millimetres",
    )
    parser.add_argument(
        "--crown-chin-target-mm",
        dest="crown_chin_target_mm",
        type=float,
        default=34.0,
        help="Target crown-to-chin distance in millimetres",
    )
    parser.add_argument(
        "--target-height-mm",
        dest="target_height_mm",
        type=float,
        default=45.0,
        help="Reference portrait height in millimetres",
    )
    parser.add_argument(
        "--min-top-mm",
        dest="min_top_mm",
        type=float,
        default=4.0,
        help="Minimum clearance above the crown in millimetres",
    )
    parser.add_argument(
        "--min-bottom-mm",
        dest="min_bottom_mm",
        type=float,
        default=8.0,
        help="Minimum clearance below the chin in millimetres",
    )
    parser.add_argument(
        "--shoulder-clearance-mm",
        dest="shoulder_clearance_mm",
        type=float,
        default=3.0,
        help="Additional clearance added below the detected shoulder line",
    )
    parser.add_argument(
        "--closed-form",
        dest="use_closed_form",
        action="store_true",
        default=True,
        help="Use the closed-form mm-budget solver for framing",
    )
    parser.add_argument(
        "--legacy-pipeline",
        dest="use_closed_form",
        action="store_false",
        help="Fallback to the legacy multi-stage padding pipeline",
    )
    parser.add_argument(
        "--no-debug",
        dest="no_debug",
        action="store_true",
        help="Disable saving debug images to disk",
    )
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
        resize_scaling=_clamp_resize_scaling(args.resize_scaling),
        square_dpi=True,
        min_top_mm=args.min_top_mm,
        min_bottom_mm=args.min_bottom_mm,
        shoulder_clearance_mm=args.shoulder_clearance_mm,
        use_closed_form=args.use_closed_form,
    )

    save_debug = not args.no_debug
    os.makedirs(args.logdir, exist_ok=True)

    print("=== Hair-Aware Passport Crop Pipeline ===")
    print(f"Input image: {args.img_path}")
    mode = "closed-form mm-budget" if params.use_closed_form else "legacy multi-stage"
    print(
        f"Params | ratio {params.target_w_over_h:.3f} | top margin {params.top_margin_ratio:.3f} | "
        f"bottom upper {params.bottom_upper_ratio:.3f} | min size {params.min_width_px}x{params.min_height_px} | "
        f"resize scaling {params.resize_scaling:.2f} | mode {mode}"
    )
    print(
        f"IRCC mm-budget | crown-chin target {params.target_crown_to_chin_mm:.1f} (range "
        f"{params.min_crown_to_chin_mm:.1f}-{params.max_crown_to_chin_mm:.1f}) | top min {params.min_top_mm:.1f} | "
        f"bottom min {params.min_bottom_mm:.1f} | shoulder clearance {params.shoulder_clearance_mm:.1f}"
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
