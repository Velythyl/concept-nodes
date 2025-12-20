import cv2
import base64
import openai
import numpy as np
from .ImageCaptioner import ImageCaptioner
import logging

log = logging.getLogger(__name__)


class OpenAICaptioner(ImageCaptioner):
    def __init__(self, model: str, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.client = openai.OpenAI()

    def encode_images(self, images: [np.ndarray]) -> [str]:
        # To RGB
        images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]
        return [
            base64.b64encode(cv2.imencode(".jpg", img)[1]).decode("utf-8")
            for img in images
        ]

    def __call__(self, images: [np.ndarray]) -> str:
        if len(images) > self.max_images:
            images = images[: self.max_images]

        base64_images = self.encode_images(images)
        response = call(self.model, base64_images, self.full_system_prompt, self.user_query, self)

        return self.postprocess_response(response)

from diskcache import Cache
cache = Cache('/tmp/openaicaptionercache') # todo cfg.paths.cache_dir)
    

@cache.memoize(ignore=(4,))
def call(model, base64_images, full_system_prompt, user_query, self):
    messages = [
        {"role": "system", "content": full_system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_query},
                *[
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_img}"
                        },
                    }
                    for base64_img in base64_images
                ],
            ],
        },
    ]
    try:
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
        )
        response = response.choices[0].message.content
    except Exception as e:
        log.warning("Error: Could not generate caption")
        response = "Invalid"

    return self.postprocess_response(response)