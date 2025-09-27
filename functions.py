import numpy as np
import cv2
from debug import put_caption, overlay_mask, colorize_grabcut_mask


def expand_box_to_hair_top(box_xyxy,
                           hair_top_y,
                           img_w,
                           img_h,
                           target_w_over_h,
                           margin_ratio,
                           ratio_tolerance=1e-3):
    """Scale a box so it keeps the target aspect ratio while ensuring the top
    edge sits ``margin_ratio`` times the final box height above the detected
    hair line.

    Args:
        box_xyxy: tuple/list ``(x1, y1, x2, y2)``.
        hair_top_y: detected top-of-hair coordinate (float or int).
        img_w, img_h: image dimensions.
        target_w_over_h: desired width/height ratio to preserve.
        margin_ratio: desired top clearance as a fraction of the final box
            height (must be in [0, 1)).
        ratio_tolerance: slack when deciding if enlargement is needed.

    Returns:
        (new_box, actual_margin_ratio)
        new_box: tuple of floats ``(nx1, ny1, nx2, ny2)``.
        actual_margin_ratio: achieved clearance expressed as a fraction of the
            final box height (``None`` if ``hair_top_y`` missing).
    """
    if hair_top_y is None:
        return tuple(float(v) for v in box_xyxy), None

    if margin_ratio < 0 or margin_ratio >= 1:
        raise ValueError("margin_ratio must be in [0, 1)")

    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    orig_height = max(1e-3, y2 - y1)
    orig_width = max(1e-3, x2 - x1)

    margin_current = float(hair_top_y) - y1
    actual_ratio = margin_current / orig_height
    if actual_ratio >= margin_ratio - ratio_tolerance:
        return (x1, y1, x2, y2), actual_ratio

    img_xmax = float(max(img_w - 1, 1))
    img_ymax = float(max(img_h - 1, 1))
    bottom = min(float(y2), img_ymax)
    dist_bottom_to_hair = bottom - float(hair_top_y)
    if dist_bottom_to_hair <= 1e-6:
        return (x1, y1, x2, y2), actual_ratio

    denom = 1.0 - float(margin_ratio)
    if denom <= 1e-6:
        raise ValueError("margin_ratio too close to 1; cannot satisfy constraint")

    required_height = dist_bottom_to_hair / denom
    new_height = max(orig_height, required_height)

    # Cannot extend beyond image top when bottom is fixed.
    max_height_at_bottom = max(1e-3, bottom)
    if new_height > max_height_at_bottom:
        new_height = max_height_at_bottom

    new_width = new_height * float(target_w_over_h)

    if new_width > img_xmax:
        scale = img_xmax / new_width
        new_width *= scale
        new_height *= scale
        if new_height < orig_height:
            new_height = orig_height
            new_width = orig_width

    cx = (x1 + x2) / 2.0
    nx1 = cx - new_width / 2.0
    nx2 = cx + new_width / 2.0

    dx_left = max(0.0, -nx1)
    dx_right = max(0.0, nx2 - img_xmax)
    nx1 += (dx_left - dx_right)
    nx2 += (dx_left - dx_right)
    nx1 = float(np.clip(nx1, 0.0, img_xmax))
    nx2 = float(np.clip(nx2, 0.0, img_xmax))
    if nx2 <= nx1:
        nx2 = min(img_xmax, nx1 + 1.0)

    ny2 = bottom
    ny1 = ny2 - new_height
    if ny1 < 0.0:
        ny1 = 0.0
        ny2 = ny1 + new_height
        if ny2 > img_ymax:
            ny2 = img_ymax
            ny1 = max(0.0, ny2 - new_height)

    new_height = max(1e-3, ny2 - ny1)
    actual_margin = max(0.0, float(hair_top_y) - ny1)
    actual_ratio = actual_margin / new_height
    return (float(nx1), float(ny1), float(nx2), float(ny2)), actual_ratio


def extend_box_downwards(box_xyxy,
                         img_w,
                         img_h,
                         upper_ratio):
    """Extend a box downward so that the original height occupies
    ``upper_ratio`` of the final height while preserving its top edge and
    horizontal center.

    Args:
        box_xyxy: tuple/list ``(x1, y1, x2, y2)``.
        img_w, img_h: image dimensions.
        upper_ratio: fraction of the final height taken by the original height
            (must be in (0, 1)).

    Returns:
        (new_box, actual_upper_ratio)
        new_box: tuple of floats ``(nx1, ny1, nx2, ny2)``.
        actual_upper_ratio: realised ratio ``orig_height / new_height``.
    """
    if upper_ratio <= 0 or upper_ratio >= 1:
        raise ValueError("upper_ratio must be in (0, 1)")

    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    w = max(1e-3, x2 - x1)
    h = max(1e-3, y2 - y1)
    if h <= 0:
        return (x1, y1, x2, y2), None

    img_xmax = float(max(img_w - 1, 1))
    img_ymax = float(max(img_h - 1, 1))

    target_scale = 1.0 / float(upper_ratio)
    cx = (x1 + x2) / 2.0

    left_space = max(0.0, cx)
    right_space = max(0.0, img_xmax - cx)
    max_scale_horizontal = max(1.0, (min(left_space, right_space) * 2.0) / w)
    max_scale_vertical = max(1.0, (img_ymax - y1) / h)

    scale = min(target_scale, max_scale_horizontal, max_scale_vertical)
    scale = max(1.0, scale)

    new_height = h * scale
    top = y1
    bottom = top + new_height
    if bottom > img_ymax:
        bottom = img_ymax
        new_height = max(1e-3, bottom - top)
        scale = new_height / h

    new_width = w * scale
    half_width = new_width / 2.0
    left = cx - half_width
    right = cx + half_width

    if left < 0.0:
        shift = -left
        left = 0.0
        right += shift
    if right > img_xmax:
        shift = right - img_xmax
        right = img_xmax
        left -= shift

    left = float(np.clip(left, 0.0, img_xmax))
    right = float(np.clip(right, 0.0, img_xmax))
    if right <= left:
        right = min(img_xmax, left + 1.0)

    new_width = right - left
    scale = new_width / w
    new_height = h * scale
    bottom = top + new_height
    if bottom > img_ymax:
        bottom = img_ymax
        new_height = max(1e-3, bottom - top)
        scale = new_height / h
        new_width = w * scale
        half_width = new_width / 2.0
        left = float(np.clip(cx - half_width, 0.0, img_xmax))
        right = float(np.clip(cx + half_width, 0.0, img_xmax))
        if right <= left:
            right = min(img_xmax, left + 1.0)
        new_width = right - left
        scale = new_width / w
        new_height = h * scale
        bottom = top + new_height

    actual_upper_ratio = h / max(new_height, 1e-3)
    return (float(left), float(top), float(right), float(bottom)), actual_upper_ratio


def detect_crown_with_retry(img,
                            face_box_xyxy,
                            img_w,
                            img_h,
                            retry_expand_ratio: float = 0.15,
                            border_tol_px: int = 2):
    """Detect the hair crown (top of head) with a single retry using an expanded ROI.

    Args:
        img: full image array.
        face_box_xyxy: tuple (x1, y1, x2, y2) describing the face box.
        img_w, img_h: image dimensions.
        retry_expand_ratio: extra fraction of face height to add upwards when retrying.
        border_tol_px: tolerance to consider the detection too close to the face box top.

    Returns:
        (y_crown_abs, debug_dict)
    """
    y_crown, debug = top_of_hair_y_debug(img, face_box_xyxy, img_w, img_h)

    need_retry = False
    if y_crown is None:
        need_retry = True
    else:
        face_top = face_box_xyxy[1]
        if y_crown >= face_top - border_tol_px:
            need_retry = True

    if not need_retry or retry_expand_ratio <= 0:
        return y_crown, debug

    x1, y1, x2, y2 = face_box_xyxy
    face_height = max(1.0, y2 - y1)
    extra = int(round(retry_expand_ratio * face_height))
    if extra <= 0:
        return y_crown, debug

    expanded_y1 = max(0, int(round(y1 - extra)))
    expanded_box = (x1, expanded_y1, x2, y2)

    y_crown_retry, debug_retry = top_of_hair_y_debug(img, expanded_box, img_w, img_h)
    if y_crown_retry is not None:
        return y_crown_retry, debug_retry

    return y_crown, debug
def adjust_box_to_ratio(x1, y1, x2, y2, img_w, img_h,
                        target_w_over_h=7/9, strategy="auto"):
    """
    Adjust an xyxy box to a target aspect ratio (w:h), preserving center.

    Args:
        x1,y1,x2,y2: box in pixels (floats ok)
        img_w, img_h: image size
        target_w_over_h: desired width/height ratio (e.g., 7/9 for portrait)
        strategy: "expand", "shrink", or "auto"
            - expand: only increase width or height
            - shrink: only decrease width or height
            - auto: change the dimension (w or h) that needs the smallest delta (may expand or shrink)

    Returns:
        nx1, ny1, nx2, ny2 (floats)
    """
    if target_w_over_h <= 0:
        raise ValueError("target_w_over_h must be > 0")

    # center + current size
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w  = max(1.0, x2 - x1)
    h  = max(1.0, y2 - y1)

    cur_ratio = w / h
    if abs(cur_ratio - target_w_over_h) < 1e-9:
        # Already at ratio; just shift/clamp to be safe
        nx1, ny1, nx2, ny2 = cx - w/2, cy - h/2, cx + w/2, cy + h/2
    else:
        # Candidate 1: change width only (w' = target * h)
        w_from_h = target_w_over_h * h
        # Candidate 2: change height only (h' = w / target)
        h_from_w = w / target_w_over_h

        if strategy == "expand":
            if cur_ratio < target_w_over_h:
                # too tall -> widen
                w = max(w, w_from_h)
            else:
                # too wide -> make taller
                h = max(h, h_from_w)
        elif strategy == "shrink":
            if cur_ratio < target_w_over_h:
                # too tall -> reduce height
                h = min(h, h_from_w)
            else:
                # too wide -> reduce width
                w = min(w, w_from_h)
        else:  # "auto": minimal change to one side
            dw = abs(w_from_h - w)
            dh = abs(h_from_w - h)
            if dw <= dh:
                w = w_from_h
            else:
                h = h_from_w

        nx1, ny1 = cx - w / 2.0, cy - h / 2.0
        nx2, ny2 = cx + w / 2.0, cy + h / 2.0

    # Shift inside image (preserve size)
    dx_left  = max(0.0, -nx1)
    dx_right = max(0.0, nx2 - img_w)
    dy_top   = max(0.0, -ny1)
    dy_bot   = max(0.0, ny2 - img_h)

    nx1 += (dx_left - dx_right); nx2 += (dx_left - dx_right)
    ny1 += (dy_top - dy_bot);    ny2 += (dy_top - dy_bot)

    # If still out (box bigger than image), scale down uniformly to fit
    new_w = nx2 - nx1
    new_h = ny2 - ny1
    if new_w > img_w or new_h > img_h:
        s = min(img_w / new_w, img_h / new_h, 1.0)
        new_w *= s; new_h *= s
        nx1 = cx - new_w / 2.0; nx2 = cx + new_w / 2.0
        ny1 = cy - new_h / 2.0; ny2 = cy + new_h / 2.0
        # shift again just in case of fp rounding
        dx_left  = max(0.0, -nx1)
        dx_right = max(0.0, nx2 - img_w)
        dy_top   = max(0.0, -ny1)
        dy_bot   = max(0.0, ny2 - img_h)
        nx1 += (dx_left - dx_right); nx2 += (dx_left - dx_right)
        ny1 += (dy_top - dy_bot);    ny2 += (dy_top - dy_bot)

    # Final safety clamp
    nx1 = float(np.clip(nx1, 0, img_w - 1))
    ny1 = float(np.clip(ny1, 0, img_h - 1))
    nx2 = float(np.clip(nx2, 0, img_w - 1))
    ny2 = float(np.clip(ny2, 0, img_h - 1))
    if nx2 <= nx1: nx2 = min(img_w - 1.0, nx1 + 1.0)
    if ny2 <= ny1: ny2 = min(img_h - 1.0, ny1 + 1.0)

    return nx1, ny1, nx2, ny2


# --- helpers for robust top-of-hair selection ---
def _remove_small_blobs(bin_img, min_area=30):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img, connectivity=8)
    out = np.zeros_like(bin_img)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 1
    return out

def _max_run_length_per_row(band_row):
    band_row = band_row.ravel()
    runs = 0
    best = 0
    for v in band_row:
        if v:
            runs += 1
            if runs > best: best = runs
        else:
            runs = 0
    return best

def find_top_row_robust(band,
                        morph_open_ks=3,
                        remove_blobs_area=30,
                        smooth_win=5,
                        coverage_thresh=0.12,
                        min_run_frac=0.18,
                        consec_rows=4):
    """
    Robustly find top row in a central band (binary FG mask):
      - morphological opening to remove speckles,
      - remove small blobs,
      - moving-average smoothing of row coverage,
      - require coverage >= coverage_thresh,
      - require max contiguous FG run >= min_run_frac*band_width,
      - require 'consec_rows' consecutive rows pass the test.
    Returns: row index (int) or None.
    """
    band = (band > 0).astype(np.uint8)

    # morphological opening to drop wisps
    if morph_open_ks and morph_open_ks > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_open_ks, morph_open_ks))
        band = cv2.morphologyEx(band, cv2.MORPH_OPEN, k, iterations=1)

    # remove tiny components
    band = _remove_small_blobs(band, min_area=int(remove_blobs_area))

    H, W = band.shape
    if H == 0 or W == 0:
        return None

    # row coverage + smoothing
    cover = band.sum(axis=1).astype(np.float32) / float(W)
    if smooth_win > 1:
        pad = smooth_win // 2
        cover = np.convolve(np.pad(cover, (pad, pad), mode='edge'),
                            np.ones(smooth_win)/smooth_win, mode='valid')

    # thresholds
    min_run = max(1, int(min_run_frac * W))
    ok = np.zeros(H, dtype=bool)
    for r in range(H):
        if cover[r] >= coverage_thresh:
            rl = _max_run_length_per_row(band[r])
            if rl >= min_run:
                ok[r] = True

    # need 'consec_rows' consecutive rows from the top
    cnt = 0
    for r in range(H):
        if ok[r]:
            cnt += 1
            if cnt >= consec_rows:
                return r - consec_rows + 1
        else:
            cnt = 0

    return None
def top_of_hair_y_debug(img, box_xyxy, img_w, img_h,
                        top_pad_ratio=1.4, side_pad_ratio=0.35,
                        below_pad_ratio=0.2,
                        band_rel_width=0.35,
                        border_bg_px=15,
                        grabcut_iters=6):
    """
    Returns:
      y_top_global (int or None),
      debug dict with:
        'roi_bgr', 'seed_mask_color', 'gc_mask_color',
        'fg_only_overlay', 'main_comp_overlay', 'band_overlay'
    """
    debug = {}
    x1, y1, x2, y2 = [int(round(v)) for v in box_xyxy]
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return None, debug

    # ROI bounds
    roi_x1 = max(0, x1 - int(side_pad_ratio * w))
    roi_x2 = min(img_w, x2 + int(side_pad_ratio * w))
    roi_y1 = max(0, y1 - int(top_pad_ratio * h))
    roi_y2 = min(img_h, y2 + int(below_pad_ratio * h))

    roi = img[roi_y1:roi_y2, roi_x1:roi_x2]
    if roi.size == 0:
        return None, debug
    rh, rw = roi.shape[:2]
    debug['roi_bgr'] = put_caption(roi, "ROI (extended above face)")

    # Seed mask
    mask = np.full((rh, rw), cv2.GC_PR_BGD, np.uint8)
    # definite BG border
    mask[:border_bg_px, :] = cv2.GC_BGD
    mask[-border_bg_px:, :] = cv2.GC_BGD
    mask[:, :border_bg_px] = cv2.GC_BGD
    mask[:, -border_bg_px:] = cv2.GC_BGD

    # Face inside ROI
    fx1, fy1 = x1 - roi_x1, y1 - roi_y1
    fx2, fy2 = x2 - roi_x1, y2 - roi_y1
    fw, fh = fx2 - fx1, fy2 - fy1
    if fw <= 0 or fh <= 0:
        return None, debug

    # Definite FG ellipse for face
    face_center = (int((fx1 + fx2) / 2), int((fy1 + fy2) / 2))
    axes = (max(1, int(0.45 * fw)), max(1, int(0.55 * fh)))
    ellipse_mask = np.zeros_like(mask, np.uint8)
    cv2.ellipse(ellipse_mask, face_center, axes, 0, 0, 360, 1, -1)
    mask[ellipse_mask == 1] = cv2.GC_FGD

    # Probable FG strip above the face to encourage hair
    hair_top_y = max(0, int(fy1 - 0.1 * fh))
    hx1 = max(0, int(fx1 + 0.08 * fw))
    hx2 = min(rw, int(fx2 - 0.08 * fw))
    hy1 = max(0, hair_top_y - int(0.5 * fh))         # push higher
    hy2 = max(0, int(fy1 - 0.02 * fh))
    if hx2 > hx1 and hy2 > hy1:
        # Don't overwrite definite BG; where mask is BG keep BG, else set PR_FGD
        region = mask[hy1:hy2, hx1:hx2]
        region = np.where(region == cv2.GC_BGD, cv2.GC_BGD, cv2.GC_PR_FGD)
        mask[hy1:hy2, hx1:hx2] = region

    debug['seed_mask_color'] = put_caption(colorize_grabcut_mask(mask), "Seed mask (BG/FG/PR_BG/PR_FG)")

    # Run GrabCut with seeds
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(roi, mask, None, bgdModel, fgdModel, grabcut_iters, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None, debug

    debug['gc_mask_color'] = put_caption(colorize_grabcut_mask(mask), "GrabCut output labels")

    # Foreground (definite or probable)
    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    debug['fg_only_overlay'] = put_caption(overlay_mask(roi, fg), "FG overlay (all FG)")

    # Keep only component intersecting center line (within a central band)
    cx_global = int((x1 + x2) / 2)
    cx_roi = int(np.clip(cx_global - roi_x1, 0, rw - 1))
    band_half = max(1, int(0.5 * band_rel_width * rw))
    bx1 = max(0, cx_roi - band_half)
    bx2 = min(rw, cx_roi + band_half)

    num_labels, labels = cv2.connectedComponents(fg, connectivity=4)
    if num_labels <= 1:
        return None, debug

    band_labels = labels[:, bx1:bx2]
    counts = np.bincount(band_labels.reshape(-1), minlength=num_labels)
    counts[0] = 0
    main_label = np.argmax(counts)
    if counts[main_label] == 0:
        return None, debug

    main_comp = (labels == main_label).astype(np.uint8)

    main_overlay = roi.copy()
    sel = main_comp.astype(bool)

    # Blend selected pixels toward white without cv2.addWeighted shape issues
    # new = 0.4 * original + 0.6 * 255
    main_overlay[sel] = (0.4 * main_overlay[sel] + 0.6 * 255).astype(np.uint8)

    # show central band
    cv2.rectangle(main_overlay, (bx1, 0), (bx2, rh - 1), (255, 0, 255), 1)
    debug['main_comp_overlay'] = put_caption(main_overlay, "Main FG component (center-band)")
    # Topmost FG row within band
    band = main_comp[:, bx1:bx2]

    y_top_in_roi = find_top_row_robust(
        band,
        morph_open_ks=3,         # 3x3 opening
        remove_blobs_area=30,    # drop tiny components
        smooth_win=5,            # moving average window
        coverage_thresh=0.12,    # ≥12% of band width must be FG
        min_run_frac=0.18,       # contiguous FG run ≥18% of band width
        consec_rows=4            # must hold for 4 consecutive rows
    )

    if y_top_in_roi is None:
        return None, debug
    y_top_global = roi_y1 + y_top_in_roi

    band_overlay = main_overlay.copy()
    cv2.line(band_overlay, (0, y_top_in_roi), (rw - 1, y_top_in_roi), (0, 0, 255), 2)
    debug['band_overlay'] = put_caption(band_overlay, "Central band + detected top line")

    return y_top_global, debug
