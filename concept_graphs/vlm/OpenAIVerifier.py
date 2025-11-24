from typing import List
import numpy as np
import logging
from tqdm import tqdm

import cv2
import base64
import openai
import json

log = logging.getLogger(__name__)

# -------------------------------------------------------------------
# JSON schema for structured outputs
# -------------------------------------------------------------------
VERIFIER_SCHEMA = {
    "type": "object",
    "properties": {
        "answers": {
            "type": "array",
            "description": "For each image, say YES if it matches the object_tag, otherwise NO.",
            "items": {
                "type": "string",
                "enum": ["YES", "NO"],
            },
            # minItems / maxItems set dynamically
        }
    },
    "required": ["answers"],
    "additionalProperties": False
}


class OpenAIVerifier:
    def __init__(
        self,
        system_prompt: str,
        user_query: str,
        max_images: int = 5,
        retry_attempts: int = 3,
        model: str = "gpt-5",
        verify_mode: str = "majority_vote",  # Options: majority_vote, unanimous, single
    ):
        self.system_prompt = system_prompt
        self.user_query = user_query
        self.max_images = max_images
        self.verify_mode = verify_mode

        self.model = model
        self.client = openai.OpenAI()
        self.retry_attempts = retry_attempts

    def encode_images(self, images: List[np.ndarray]) -> List[str]:
        images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]
        return [
            base64.b64encode(cv2.imencode(".jpg", img)[1]).decode("utf-8")
            for img in images
        ]
    
    def postprocess_response(self, verification_list: List[str]) -> bool:
        yes_count = verification_list.count("YES")
        no_count = verification_list.count("NO")

        if self.verify_mode == "majority_vote":
            return yes_count > no_count
        elif self.verify_mode == "unanimous":
            return yes_count == len(verification_list)
        elif self.verify_mode == "single":
            return yes_count > 0
        else:
            log.error(f"Unknown verify_mode: {self.verify_mode}")
            return False
        
    
    def __call__(self, images: List[np.ndarray], object_tag: str) -> bool:
        if len(images) > self.max_images:
            images = images[: self.max_images]

        base64_images = self.encode_images(images)
        messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "You are a strict visual verifier. "
                            "For each image, output YES if it clearly contains the object "
                            "corresponding to the object tag, otherwise output NO. "
                            f"Object tag: {object_tag}"
                        ),
                    },
                    *[
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{base64_img}",
                        }
                        for base64_img in base64_images
                    ],
                ],
            },
        ]

        # Configure array length to match the number of images
        VERIFIER_SCHEMA["properties"]["answers"]["minItems"] = len(images)
        VERIFIER_SCHEMA["properties"]["answers"]["maxItems"] = len(images)

        for attempt in range(self.retry_attempts):
            try:
                response = self.client.responses.create(
                    model=self.model,
                    input=messages,
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "verification_response",
                            "schema": VERIFIER_SCHEMA,
                            "strict": True,
                        },
                    }
                )

                result_raw = response.output_text
                result = json.loads(result_raw)["answers"]

                if len(result) != len(images):
                    raise ValueError(
                        f"Expected {len(images)} items, got {len(result)}."
                    )   
                
                return self.postprocess_response(result)

            except Exception as e:
                log.error(
                    f"OpenAI API call or postprocessing failed on attempt {attempt + 1}: {e}"
                )

        # If all retries failed, log and return False
        log.error(f"All {self.retry_attempts} attempts to call OpenAI API failed.")
        return False