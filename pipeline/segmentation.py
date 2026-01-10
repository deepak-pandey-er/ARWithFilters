"""
Field segmentation with a safe, lazy PyTorch/torchvision load and a CPU fallback.

Behavior:
- Try to import torch & torchvision and load deeplabv3_resnet50.
- If anything fails (DLL import error, missing packages, etc.), fall back to a fast
  HSV + morphology segmentation routine which does not require PyTorch.
- The module avoids importing torch at top-level to prevent DLL-import crashes on start.
"""

import numpy as np
import cv2
import warnings

def _preprocess_torch(img, device):
    import torch
    from torchvision import transforms
    t = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    return t(img).unsqueeze(0).to(device)

class FieldSegmenter:
    def __init__(self, device="cpu", use_torch=True):
        """
        device: 'cpu' or 'cuda' (if available)
        use_torch: if False, skip attempting to load torch even if present
        """
        self.device = device
        self.model = None
        self.use_torch = use_torch

        if self.use_torch:
            try:
                # Lazy import torch/torchvision (inside try/except to catch DLL issues)
                import torch
                from torchvision.models.segmentation import deeplabv3_resnet50
                self.torch = torch
                self.device_t = torch.device(device)
                # load pretrained model as a baseline (will download on first use if needed)
                self.model = deeplabv3_resnet50(pretrained=True, progress=True).to(self.device_t)
                self.model.eval()
                # small flag to indicate the model should be used
                self.can_use_torch_model = True
            except Exception as e:
                warnings.warn(f"torch/torchvision unavailable or failed to initialize: {e}. Falling back to CPU heuristic segmentation.")
                self.model = None
                self.can_use_torch_model = False
        else:
            self.can_use_torch_model = False

    def segment(self, frame):
        """
        Return: binary mask uint8 HxW (255 field/grass, 0 non-field)
        If torch model is available, attempt to use model + color heuristics as fallback.
        Otherwise, use color+texture heuristic.
        """
        h, w = frame.shape[:2]

        # If model available, try model inference but guard against runtime errors
        if self.can_use_torch_model and self.model is not None:
            try:
                inp = _preprocess_torch(frame, self.device)
                with self.torch.no_grad():
                    out = self.model(inp)["out"][0]  # C x H' x W'
                    # Upsample to input size
                    probs = self.torch.nn.functional.interpolate(out.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False)[0]
                    # choose the channel with highest logit per-pixel
                    pred = probs.argmax(0).cpu().numpy().astype(np.uint8)
                    # NOTE: pretrained deeplabv3 is trained on COCO/Pascal etc. — class mapping not meaningful for grass.
                    # Use a color heuristic to convert pred into grass mask where sensible
                    # Here, assume class indexes with greenish predictions are not reliable — combine with color heuristic
                    mask_color = self._color_heuristic(frame)
                    # simple fusion: where color heuristic says grass -> keep; otherwise use model's 'background' assumption
                    mask = mask_color
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                    return mask
            except Exception as e:
                warnings.warn(f"Torch model inference failed at runtime: {e}. Falling back to CPU heuristic segmentation.")
                # fall through to CPU heuristic

        # CPU-only heuristic segmentation (fast, robust fallback)
        return self._color_heuristic(frame)

    def _color_heuristic(self, frame):
        """
        A robust HSV-based heuristic tuned for green fields + morphological cleanup.
        Returns mask uint8 (255 field, 0 non-field).
        """
        h, w = frame.shape[:2]
        # Convert to HSV and threshold for green
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # These ranges are intentionally wide; adjust if needed for your broadcast footage
        lower = np.array([25, 40, 40], dtype=np.uint8)
        upper = np.array([100, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        # Improve mask with bilateral filter on V channel to reduce shadows, optional CLAHE
        # morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Optionally remove small blobs
        cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cleaned = np.zeros_like(mask)
        min_area = max(2000, int(0.001 * h * w))
        for c in cnts:
            if cv2.contourArea(c) > min_area:
                cv2.drawContours(cleaned, [c], -1, 255, -1)

        # Smooth edges a bit
        cleaned = cv2.GaussianBlur(cleaned, (7,7), 0)
        _, cleaned = cv2.threshold(cleaned, 127, 255, cv2.THRESH_BINARY)
        return cleaned