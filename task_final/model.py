import os
from typing import Dict, List, Optional
from uuid import uuid4

from label_studio_converter import brush
from label_studio_ml.model import LabelStudioMLBase

from sam_predictor import SAMPredictor


SAM_CHOICE = os.getenv("SAM_CHOICE", "MobileSAM")
PREDICTOR = SAMPredictor(SAM_CHOICE)


class RiverSAMSegmenter(LabelStudioMLBase):

    def setup(self):
        self.set("model_version", f"{self.__class__.__name__}-v1")

    def predict(
        self,
        tasks: List[Dict],
        context: Optional[Dict] = None,
        **kwargs,
    ) -> List[Dict]:
        if not context or not context.get("result"):
            return [
                {
                    "result": [],
                    "model_version": self.get("model_version"),
                    "score": 0.0,
                }
            ]

        from_name, to_name, image_value = self.get_first_tag_occurence(
            "BrushLabels",
            "Image",
        )

        first_ctx = context["result"][0]
        image_width = first_ctx.get("original_width") or first_ctx.get(
            "original_image_width"
        )
        image_height = first_ctx.get("original_height") or first_ctx.get(
            "original_image_height"
        )

        point_coords = []
        point_labels = []
        input_box = None
        selected_label = "river_water"

        for ctx in context["result"]:
            value = ctx.get("value", {})
            ctx_type = ctx.get("type")
            x = value.get("x", 0) * image_width / 100
            y = value.get("y", 0) * image_height / 100

            labels = value.get(ctx_type) or value.get("labels") or [selected_label]
            selected_label = labels[0]

            if ctx_type == "keypointlabels":
                point_coords.append([int(x), int(y)])
                point_labels.append(1 if ctx.get("is_positive", True) else 0)

            if ctx_type == "rectanglelabels":
                box_width = value.get("width", 0) * image_width / 100
                box_height = value.get("height", 0) * image_height / 100
                input_box = [
                    int(x),
                    int(y),
                    int(x + box_width),
                    int(y + box_height),
                ]

        img_path = tasks[0]["data"][image_value]
        predictor_results = PREDICTOR.predict(
            img_path=img_path,
            point_coords=point_coords or None,
            point_labels=point_labels or None,
            input_box=input_box,
            task=tasks[0],
            path_resolver=self.get_local_path,
        )

        return self._format_results(
            masks=predictor_results["masks"],
            probs=predictor_results["probs"],
            width=image_width,
            height=image_height,
            from_name=from_name,
            to_name=to_name,
            label=selected_label,
        )

    def _format_results(self, masks, probs, width, height, from_name, to_name, label):
        results = []
        total_prob = 0

        for mask, prob in zip(masks, probs):
            label_id = str(uuid4())[:8]
            rle = brush.mask2rle(mask.astype("uint8") * 255)
            total_prob += prob

            results.append(
                {
                    "id": label_id,
                    "from_name": from_name,
                    "to_name": to_name,
                    "original_width": width,
                    "original_height": height,
                    "image_rotation": 0,
                    "value": {
                        "format": "rle",
                        "rle": rle,
                        "brushlabels": [label],
                    },
                    "score": prob,
                    "type": "brushlabels",
                    "readonly": False,
                }
            )

        return [
            {
                "result": results,
                "model_version": self.get("model_version"),
                "score": total_prob / max(len(results), 1),
            }
        ]
