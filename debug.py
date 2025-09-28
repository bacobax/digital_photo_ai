# Cell 0: helpers + imports
import cv2
import numpy as np

def colorize_grabcut_mask(mask):
    """
    Color map for GrabCut labels:
      0: GC_BGD      -> red-ish
      1: GC_FGD      -> green
      2: GC_PR_BGD   -> orange
      3: GC_PR_FGD   -> cyan
    """
    h, w = mask.shape[:2]
    out = np.zeros((h, w, 3), np.uint8)
    out[mask == cv2.GC_BGD]    = (0,   0, 200)   # BGR
    out[mask == cv2.GC_FGD]    = (0, 200,   0)
    out[mask == cv2.GC_PR_BGD] = (0, 165, 255)
    out[mask == cv2.GC_PR_FGD] = (255, 255, 0)
    return out

def overlay_mask(base, mask_bin, alpha=0.5):
    """Overlay a white binary mask on base image."""
    base = base.copy()
    white = np.full_like(base, 255)
    sel = mask_bin > 0
    base[sel] = cv2.addWeighted(base[sel], 1 - alpha, white[sel], alpha, 0)
    return base

def put_caption(img, text):
    """Add a black bar caption at the top of an image."""
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    return img

# Where to save the debug images:
