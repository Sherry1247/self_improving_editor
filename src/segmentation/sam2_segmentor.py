"""SAM2-based object segmentor implementation."""

import logging
from typing import List, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image

from .base import Segmentor

logger = logging.getLogger("closed_loop_editor")


class SAM2Segmentor(Segmentor):
    """
    SAM2-based Segmentor that uses bounding boxes as prompts.
    
    Supports real Hugging Face SAM2 model execution and a deterministic
    mock fallback that generates ellipsoidal masks matching the bounding boxes.
    """

    def __init__(
        self,
        model_id: str = "facebook/sam2.1-hiera-small",
        device: str = "cpu",
        use_mock: bool = False,
    ):
        self.model_id = model_id
        self.device = device
        self.use_mock = use_mock

        self.model = None
        self.processor = None

        if not self.use_mock:
            try:
                from transformers import Sam2Model, Sam2Processor
                logger.info("Loading SAM2 model: %s on %s", model_id, device)
                self.processor = Sam2Processor.from_pretrained(model_id)
                self.model = Sam2Model.from_pretrained(model_id).to(device)
                self.model.eval()
            except Exception as e:
                logger.warning(
                    "Failed to load real SAM2 model (%s). Falling back to MOCK mode.",
                    e
                )
                self.use_mock = True

    def segment(
        self,
        image: Union[np.ndarray, Image.Image],
        bounding_boxes: List[Tuple[int, int, int, int]],
    ) -> List[np.ndarray]:
        """
        Generate binary segmentation masks for provided bounding boxes.

        Args:
            image: OpenCV BGR array or PIL Image.
            bounding_boxes: List of bounding boxes as (x1, y1, x2, y2).

        Returns:
            List of binary masks as boolean numpy arrays of shape (H, W).
        """
        # Read image size
        if isinstance(image, np.ndarray):
            h, w = image.shape[0], image.shape[1]
            pil_image = self._to_pil(image)
        else:
            w, h = image.size
            pil_image = image

        if not bounding_boxes:
            return []

        if self.use_mock:
            return self._mock_segment(h, w, bounding_boxes)

        try:
            masks = []
            # We process bounding boxes one by one or in a batch.
            # Processing one by one is very robust against indexing issues.
            for box in bounding_boxes:
                # input_boxes expects a list of lists of lists: [[[x1, y1, x2, y2]]]
                inputs = self.processor(
                    pil_image,
                    input_boxes=[[[box[0], box[1], box[2], box[3]]]],
                    return_tensors="pt"
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)

                # Post-process masks
                post_processed = self.processor.post_process_masks(
                    outputs.pred_masks,
                    inputs["original_sizes"],
                    inputs["reshaped_input_sizes"]
                )

                # post_processed[0] shape: (1, 1, num_masks, H, W)
                pred_masks = post_processed[0][0, 0]  # shape: (num_masks, H, W)
                iou_scores = outputs.iou_predictions[0, 0]  # shape: (num_masks,)

                # Pick the mask with the highest predicted IoU score
                best_idx = torch.argmax(iou_scores).item()
                best_mask = pred_masks[best_idx].cpu().numpy()  # boolean array shape (H, W)
                masks.append(best_mask)

            return masks

        except Exception as e:
            logger.error("Error running real SAM2: %s. Falling back to mock.", e)
            return self._mock_segment(h, w, bounding_boxes)

    def _mock_segment(
        self,
        height: int,
        width: int,
        bounding_boxes: List[Tuple[int, int, int, int]],
    ) -> List[np.ndarray]:
        """Generate high-quality mock oval masks matching the bounding boxes."""
        masks = []
        for box in bounding_boxes:
            x1, y1, x2, y2 = box
            # Create empty black mask
            mask = np.zeros((height, width), dtype=np.uint8)
            
            # Clip bounding box to image boundaries
            x1_c = max(0, min(width - 1, x1))
            y1_c = max(0, min(height - 1, y1))
            x2_c = max(0, min(width - 1, x2))
            y2_c = max(0, min(height - 1, y2))
            
            if x2_c > x1_c and y2_c > y1_c:
                # Compute ellipse center and axes
                center = ((x1_c + x2_c) // 2, (y1_c + y2_c) // 2)
                axes = ((x2_c - x1_c) // 2, (y2_c - y1_c) // 2)
                # Draw a filled ellipse on the mask
                cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
                
            masks.append(mask > 0)
            
        return masks

    def _to_pil(self, image: np.ndarray) -> Image.Image:
        """Convert BGR numpy array to PIL RGB."""
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
