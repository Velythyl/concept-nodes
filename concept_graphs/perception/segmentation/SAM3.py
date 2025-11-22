from typing import Tuple, List
import numpy as np
from PIL import Image

import warnings

# Filter out user warnings from a specific package
warnings.filterwarnings("ignore", category=UserWarning, module="sam3")

import torch

from sam3.train.data.collator import collate_fn_api as collate
from sam3.model.utils.misc import copy_data_to_device
from sam3 import build_sam3_image_model
from sam3.train.data.sam3_image_dataset import (
    InferenceMetadata,
    FindQueryLoaded,
    Image as SAMImage,
    Datapoint,
)
from sam3.train.transforms.basic_for_api import (
    ComposeAPI,
    RandomResizeAPI,
    ToTensorAPI,
    NormalizeAPI,
)
from sam3.eval.postprocessors import PostProcessImage


from .SegmentationModel import SegmentationModel


class SAM3(SegmentationModel):
    def __init__(
        self,
        model_type: str,
        bpe_path: str,
        queries_path: str,
        threshold: float = 0.5,
        sam_batch_size: int = 10,
        device: str = "cuda",
    ):
        self.counter = 0
        self.model_type = model_type
        self.bpe_path = bpe_path
        self.queries_path = queries_path
        self.device = device
        self.model = build_sam3_image_model(bpe_path=bpe_path)
        self.transform = ComposeAPI(
            transforms=[
                RandomResizeAPI(
                    sizes=1008, max_size=1008, square=True, consistent_transform=False
                ),
                ToTensorAPI(),
                NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.postprocessor = PostProcessImage(
            max_dets_per_img=-1,  # if this number is positive, the processor will return topk. For this demo we instead limit by confidence, see below
            iou_type="segm",  # we want masks
            use_original_sizes_box=True,  # boxes should be resized to the image size
            use_original_sizes_mask=True,  # masks should be resized to the image size
            convert_mask_to_rle=False,  # the postprocessor supports efficient conversion to RLE format
            detection_threshold=threshold,  # Only return confident detections
            to_cpu=True,
        )
        self.prompts = self.load_text_prompts(queries_path)
        self.batch_size = sam_batch_size

    def create_empty_datapoint(self) -> Datapoint:
        """A datapoint is a single image on which we can apply several queries at once."""
        return Datapoint(find_queries=[], images=[])

    def set_image(self, datapoint: Datapoint, pil_image: Image.Image):
        """Add the image to be processed to the datapoint"""
        w, h = pil_image.size
        datapoint.images = [SAMImage(data=pil_image, objects=[], size=[h, w])]

    def add_text_prompt(self, datapoint: Datapoint, text_query: str) -> int:
        """Add a text query to the datapoint"""

        w, h = datapoint.images[0].size
        datapoint.find_queries.append(
            FindQueryLoaded(
                query_text=text_query,
                image_id=0,
                object_ids_output=[],  # unused for inference
                is_exhaustive=True,  # unused for inference
                query_processing_order=0,
                inference_metadata=InferenceMetadata(
                    coco_image_id=self.counter,
                    original_image_id=self.counter,
                    original_category_id=1,
                    original_size=[w, h],
                    object_id=0,
                    frame_index=0,
                ),
            )
        )
        self.counter += 1
        return self.counter - 1

    def load_text_prompts(self, queries_path: str) -> List[str]:
        """Load text prompts from a file each line is a prompt"""
        with open(queries_path, "r") as f:
            prompts = f.read().splitlines()
        return prompts

    def __call__(self, img: np.ndarray):
        with torch.inference_mode(), torch.no_grad():
            pil_img = Image.fromarray(img)
            all_prompts = self.prompts

            masks_list, boxes_list, scores_list = [], [], []

            if self.batch_size == -1:
                self.batch_size = len(all_prompts)

            for start in range(0, len(all_prompts), self.batch_size):
                prompt_batch = all_prompts[start : start + self.batch_size]

                datapoint = self.create_empty_datapoint()
                self.set_image(datapoint, pil_img)

                for prompt in prompt_batch:
                    self.add_text_prompt(datapoint, prompt)

                datapoint = self.transform(datapoint)
                batch = collate([datapoint], dict_key="sam3_image")["sam3_image"]
                batch = copy_data_to_device(batch, self.device, non_blocking=True)

                outputs = self.model(batch)
                results = self.postprocessor.process_results(
                    outputs, batch.find_metadatas
                )

                for _, processed in results.items():
                    masks_list.append(processed["masks"].squeeze(1).cpu())
                    boxes_list.append(processed["boxes"].cpu())
                    scores_list.append(processed["scores"].cpu())

                del batch, outputs, results, datapoint
                torch.cuda.empty_cache()

            masks = torch.cat(masks_list, dim=0)
            boxes = torch.cat(boxes_list, dim=0)
            scores = torch.cat(scores_list, dim=0)

            boxes = boxes.round().int()

            self.counter = 0
            return masks, boxes, scores
