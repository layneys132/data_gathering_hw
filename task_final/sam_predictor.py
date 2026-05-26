import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from label_studio_ml.utils import InMemoryLRUDictCache


logger = logging.getLogger(__name__)
BACKEND_DIR = Path(__file__).resolve().parent
MODELS_DIR = BACKEND_DIR / "models"


def checkpoint_path(env_name: str, default_name: str) -> str:
    raw = os.getenv(env_name)
    path = Path(raw) if raw else MODELS_DIR / default_name
    if not path.is_absolute():
        path = BACKEND_DIR / path
    return str(path)


MOBILESAM_CHECKPOINT = checkpoint_path("MOBILESAM_CHECKPOINT", "mobile_sam.pt")
SAM_CHECKPOINT = checkpoint_path("VITH_CHECKPOINT", "sam_vit_h_4b8939.pth")


class SAMPredictor:
    def __init__(self, model_choice: str):
        self.model_choice = model_choice
        self.cache = InMemoryLRUDictCache(1)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if model_choice == "MobileSAM":
            from mobile_sam import SamPredictor, sam_model_registry

            checkpoint = MOBILESAM_CHECKPOINT
            registry_key = "vit_t"
        else:
            raise ValueError("SAM_CHOICE must be MobileSAM")


        logger.info(f"Loading {model_choice} checkpoint from {checkpoint}")
        sam = sam_model_registry[registry_key](checkpoint=checkpoint)
        sam.to(device=self.device)
        self.predictor = SamPredictor(sam)

    def set_image(self, img_path: str, task: Optional[Dict], path_resolver):
        payload = self.cache.get(img_path)
        if payload is not None:
            return payload

        if path_resolver is None:
            raise ValueError("path_resolver is required")

        task_id = task.get("id") if task else None
        image_path = path_resolver(img_path, task_id=task_id)
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.ascontiguousarray(image, dtype=np.uint8)
        self.predictor.set_image(image)

        payload = {"image_shape": image.shape[:2]}
        self.cache.put(img_path, payload)
        return payload

    def predict(
        self,
        img_path: str,
        point_coords: Optional[List[List]] = None,
        point_labels: Optional[List] = None,
        input_box: Optional[List] = None,
        task: Optional[Dict] = None,
        path_resolver=None,
    ):
        self.set_image(img_path, task=task, path_resolver=path_resolver)

        coords = np.array(point_coords, dtype=np.float32) if point_coords else None
        labels = np.array(point_labels, dtype=np.float32) if point_labels else None
        box = np.array(input_box, dtype=np.float32) if input_box else None

        masks, probs, _ = self.predictor.predict(
            point_coords=coords,
            point_labels=labels,
            box=box,
            multimask_output=False,
        )

        return {
            "masks": [masks[0].astype(np.uint8)],
            "probs": [float(probs[0])],
        }