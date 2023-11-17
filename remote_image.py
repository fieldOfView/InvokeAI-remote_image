import cv2
import numpy as np
import urllib
import requests
import os

from PIL import Image
from invokeai.app.invocations.primitives import (
    ImageField,
    ImageOutput,
)
from invokeai.app.services.image_records.image_records_common import (
    ImageCategory,
    ResourceOrigin,
)
from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    InputField,
    InvocationContext,
    invocation,
)


@invocation(
    "LoadRemoteImageInvocation",
    title="Load Remote Image",
    tags=["image", "load", "remote", "get", "url"],
    category="image",
    version="0.1.0",
)
class LoadRemoteImageInvocation(BaseInvocation):
    """Load an image from a remote URL and provide it as output."""

    # Inputs
    image_url: str = InputField(description="The URL of the image to get")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        # Download and decode the image from the remote URL

        with urllib.request.urlopen(self.image_url) as response:
            contents = np.asarray(bytearray(response.read()), dtype="uint8")

        if not response:
            raise Exception(f"Failed to retreive a file from URL {self.image_url}")

        # Load the image into PIL Image
        try:
            image = Image.fromarray(
                cv2.cvtColor(
                    cv2.imdecode(contents, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
                )
            )
        except cv2.error:
            raise Exception(f"Failed to decode image from URL {self.image_url}")

        # Create the ImageField object
        image_dto = context.services.images.create(
            image=image,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate,
        )

        return ImageOutput(
            image=ImageField(image_name=image_dto.image_name),
            width=image_dto.width,
            height=image_dto.height,
        )


@invocation(
    "PostImageToRemoteInvocation",
    title="Post Image to Remote Server",
    tags=["image", "send", "remote", "post", "url"],
    category="image",
    version="0.1.0",
)
class PostImageToRemoteInvocation(BaseInvocation):
    """Post an image to a remote URL."""

    # Inputs
    image: ImageField = InputField(description="The image to 'POST'")
    endpoint: str = InputField(description="The URL of endpoint to POST the image to")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image_path = context.services.images.get_path(self.image.image_name)

        with open(image_path, "rb") as image_file:
            image_name = os.path.basename(image_path)
            files = {
                "image": (
                    image_name,
                    image_file,
                    "multipart/form-data",
                    {"Expires": "0"},
                )
            }
            response = requests.post(self.endpoint, files=files)

            if response.status_code not in [200, 201]:
                raise Exception(
                    f"Failed to post the image to endpoint {self.endpoint} with return code {response.status_code}"
                )

        image = context.services.images.get_pil_image(self.image.image_name)

        return ImageOutput(
            image=ImageField(image_name=self.image.image_name),
            width=image.width,
            height=image.height,
        )
