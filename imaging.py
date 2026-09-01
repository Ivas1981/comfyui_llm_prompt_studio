"""Image tensor to base64 JPEG conversion for vision-model requests."""
import base64
import io
import logging

import numpy as np
from PIL import Image

logger = logging.getLogger("llm_prompt_studio")

__all__ = ["image_to_base64"]


def image_to_base64(image_tensor, max_size: int = 1024) -> str:
    if image_tensor.shape[0] > 1:
        logger.warning(
            "image_to_base64: only the first image of the batch (%d frames) is sent "
            "to the vision model; the rest are ignored.", image_tensor.shape[0])
    arr = image_tensor[0].cpu().numpy()
    img = Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8))
    img = img.convert("RGB")
    img.thumbnail((max_size, max_size))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()