##
##
## Stablender Diffusion Load Script- For loading stability-sdk into blender
## version: 0.0.0
## author: Shellworld (twitter: @shellworld1; github.com/shellward)
##
##


import pathlib
import sys
import os
import uuid
import random
import io
import logging
import time
import mimetypes
import pip

#install pip packages

pip.main(['install', 'grpcio', '--user'])
pip.main(['install', 'python-dotenv', '--user'])
pip.main(['install', 'stability-sdk', '--user'])
pip.main(['install', 'grpcio-tools', '--user'])
pip.main(['install', 'PIL', '--user'])

#REPLACE THIS WITH YOUR OWN PATH- you'll need to figure out which version of python blender is running, then locate the site-packages directory.
#This is not necessarily attached to your local python installation- for example in windows 11 it's in your roaming app data/python/py version directory.
#This might be a little different in linux/mac- but you are looking for the python-site-packages directory to attach to the syspath

PACKAGES_PATH = "C:\\Users\\Henry\\AppData\\Roaming\\Python\\Python310\\site-packages"

## REPLACE THIS WITH YOUR OWN API KEY, which can be found at https://beta.dreamstudio.ai/membership
STABILITY_API_KEY = "YOUR API KEY"
#Set the default size for your requested image
size=(512,512)



#Insert paths
packages_path= PACKAGES_PATH
sys.path.insert(0, packages_path )

interfaces_path = f"{packages_path}\\stability_sdk"
sys.path.insert(1,interfaces_path)

generation_path = f"{interfaces_path}\\interfaces\\gooseai\\generation"
sys.path.insert(2,generation_path)

#Import the stability_sdk
import grpc
from argparse import ArgumentParser, Namespace
from typing import Dict, Generator, List, Union, Any, Sequence, Tuple
from dotenv import load_dotenv
from google.protobuf.json_format import MessageToJson


#This is all from the client.py file in the stability-sdk
load_dotenv()

thisPath = pathlib.Path(__file__).parent.resolve()
genPath = thisPath / "interfaces/gooseai/generation"
sys.path.append(str(genPath))

import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
import stability_sdk.interfaces.gooseai.generation.generation_pb2_grpc as generation_grpc

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

algorithms: Dict[str, int] = {
    "ddim": generation.SAMPLER_DDIM,
    "plms": generation.SAMPLER_DDPM,
    "k_euler": generation.SAMPLER_K_EULER,
    "k_euler_ancestral": generation.SAMPLER_K_EULER_ANCESTRAL,
    "k_heun": generation.SAMPLER_K_HEUN,
    "k_dpm_2": generation.SAMPLER_K_DPM_2,
    "k_dpm_2_ancestral": generation.SAMPLER_K_DPM_2_ANCESTRAL,
    "k_lms": generation.SAMPLER_K_LMS,
}


def get_sampler_from_str(s: str) -> generation.DiffusionSampler:
    """
    Convert a string to a DiffusionSampler enum.

    :param s: The string to convert.
    :return: The DiffusionSampler enum.
    """
    algorithm_key = s.lower().strip()
    algorithm = algorithms.get(algorithm_key, None)
    if algorithm is None:
        raise ValueError(f"unknown sampler {s}")
    return algorithm


def process_artifacts_from_answers(
    prefix: str,
    answers: Union[
        Generator[generation.Answer, None, None], Sequence[generation.Answer]
    ],
    write: bool = True,
    verbose: bool = False,
) -> Generator[Tuple[str, generation.Artifact], None, None]:
    """
    Process the Artifacts from the Answers.

    :param prefix: The prefix for the artifact filenames.
    :param answers: The Answers to process.
    :param write: Whether to write the artifacts to disk.
    :param verbose: Whether to print the artifact filenames.
    :return: A Generator of tuples of artifact filenames and Artifacts, intended
        for passthrough.
    """
    idx = 0
    for resp in answers:
        for artifact in resp.artifacts:
            artifact_p = f"{prefix}-{resp.request_id}-{resp.answer_id}-{idx}"
            if artifact.type == generation.ARTIFACT_IMAGE:
                ext = mimetypes.guess_extension(artifact.mime)
                contents = artifact.binary
            elif artifact.type == generation.ARTIFACT_CLASSIFICATIONS:
                ext = ".pb.json"
                contents = MessageToJson(artifact.classifier).encode("utf-8")
            elif artifact.type == generation.ARTIFACT_TEXT:
                ext = ".pb.json"
                contents = MessageToJson(artifact).encode("utf-8")
            else:
                ext = ".pb"
                contents = artifact.SerializeToString()
            out_p = f"{artifact_p}{ext}"
            if write:
                with open(out_p, "wb") as f:
                    f.write(bytes(contents))
                    if verbose:
                        artifact_t = generation.ArtifactType.Name(artifact.type)
                        logger.info(f"wrote {artifact_t} to {out_p}")

            yield [out_p, artifact]
            idx += 1


def open_images(
    images: Union[
        Sequence[Tuple[str, generation.Artifact]],
        Generator[Tuple[str, generation.Artifact], None, None],
    ],
    verbose: bool = False,
) -> Generator[Tuple[str, generation.Artifact], None, None]:
    """
    Open the images from the filenames and Artifacts tuples.

    :param images: The tuples of Artifacts and associated images to open.
    :return:  A Generator of tuples of image filenames and Artifacts, intended
     for passthrough.
    """
    from PIL import Image

    for path, artifact in images:
        if artifact.type == generation.ARTIFACT_IMAGE:
            if verbose:
                logger.info(f"opening {path}")
            img = Image.open(io.BytesIO(artifact.binary))
            img.show()
        yield [path, artifact]


class StabilityInference:
    def __init__(
        self,
        host: str = "grpc.stability.ai:443",
        key: str = f"{STABILITY_API_KEY}",
        engine: str = "stable-diffusion-v1",
        verbose: bool = False,
        wait_for_ready: bool = True,
    ):
        """
        Initialize the client.

        :param host: Host to connect to.
        :param key: Key to use for authentication.
        :param engine: Engine to use.
        :param verbose: Whether to print debug messages.
        :param wait_for_ready: Whether to wait for the server to be ready, or
            to fail immediately.
        """
        self.verbose = verbose
        self.engine = engine

        self.grpc_args = {"wait_for_ready": wait_for_ready}

        if verbose:
            logger.info(f"Opening channel to {host}")

        call_credentials = []

        if host.endswith("443"):
            if key:
                call_credentials.append(
                    grpc.access_token_call_credentials(f"{key}"))
            else:
                raise ValueError(f"key is required for {host}")
            channel_credentials = grpc.composite_channel_credentials(
                grpc.ssl_channel_credentials(), *call_credentials
            )
            channel = grpc.secure_channel(host, channel_credentials)
        else:
            if key:
                logger.warning(
                    "Not using authentication token due to non-secure transport"
                )
            channel = grpc.insecure_channel(host)

        if verbose:
            logger.info(f"Channel opened to {host}")
        self.stub = generation_grpc.GenerationServiceStub(channel)

    def generate(
        self,
        prompt: Union[List[str], str],
        height: int = 512,
        width: int = 512,
        cfg_scale: float = 7.0,
        sampler: generation.DiffusionSampler = generation.SAMPLER_K_LMS,
        steps: int = 50,
        seed: Union[Sequence[int], int] = 0,
        samples: int = 1,
        safety: bool = True,
        classifiers: generation.ClassifierParameters = None,
    ) -> Generator[generation.Answer, None, None]:
        """
        Generate images from a prompt.

        :param prompt: Prompt to generate images from.
        :param height: Height of the generated images.
        :param width: Width of the generated images.
        :param cfg_scale: Scale of the configuration.
        :param sampler: Sampler to use.
        :param steps: Number of steps to take.
        :param seed: Seed for the random number generator.
        :param samples: Number of samples to generate.
        :param safety: Whether to use safety mode.
        :param classifications: Classifier parameters to use.
        :return: Generator of Answer objects.
        """
        if safety and classifiers is None:
            classifiers = generation.ClassifierParameters()

        if not prompt:
            raise ValueError("prompt must be provided")

        request_id = str(uuid.uuid4())

        if not seed:
            seed = [random.randrange(0, 4294967295)]

        if isinstance(prompt, str):
            prompt = [generation.Prompt(text=prompt)]
        else:
            prompt = [generation.Prompt(text=p) for p in prompt]

        rq = generation.Request(
            engine_id=self.engine,
            request_id=request_id,
            prompt=prompt,
            image=generation.ImageParameters(
                transform=generation.TransformType(diffusion=sampler),
                height=height,
                width=width,
                seed=seed,
                steps=steps,
                samples=samples,
                parameters=[
                    generation.StepParameter(
                        scaled_step=0,
                        sampler=generation.SamplerParameters(cfg_scale=cfg_scale),
                    )
                ],
            ),
            classifier=classifiers,
        )

        if self.verbose:
            logger.info("Sending request.")

        start = time.time()
        for answer in self.stub.Generate(rq, **self.grpc_args):
            duration = time.time() - start
            if self.verbose:
                if len(answer.artifacts) > 0:
                    artifact_ts = [
                        generation.ArtifactType.Name(artifact.type)
                        for artifact in answer.artifacts
                    ]
                    logger.info(
                        f"Got {answer.answer_id} with {artifact_ts} in "
                        f"{duration:0.2f}s"
                    )
                else:
                    logger.info(
                        f"Got keepalive {answer.answer_id} in "
                        f"{duration:0.2f}s"
                    )

            yield answer
            start = time.time()


def build_request_dict(cli_args: Namespace) -> Dict[str, Any]:
    """
    Build a Request arguments dictionary from the CLI arguments.
    """
    return {
        "height": cli_args.height,
        "width": cli_args.width,
        "cfg_scale": cli_args.cfg_scale,
        "sampler": get_sampler_from_str(cli_args.sampler),
        "steps": cli_args.steps,
        "seed": cli_args.seed,
        "samples": cli_args.num_samples,
    }

#import bpy
import bpy
import numpy as np

from PIL import Image
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
import base64
import zlib
import struct

#You will also need to put your API key here
stability_api = client.StabilityInference(
    key=f"{STABILITY_API_KEY}", 
    verbose=True,
)



def pil_to_image(pil_image, name='NewImage'):
    '''
    PIL image pixels is 2D array of byte tuple (when mode is 'RGB', 'RGBA') or byte (when mode is 'L')
    bpy image pixels is flat array of normalized values in RGBA order
    '''
    # setup PIL image reading
    width = pil_image.width
    height = pil_image.height
    pil_pixels = pil_image.load()
    byte_to_normalized = 1.0 / 255.0
    num_pixels = width * height
    # setup bpy image
    channels = 4
    bpy_image = bpy.data.images.new(name, width=width, height=height)
    # bpy image has a flat RGBA array (similar to JS Canvas)
    bpy_image.pixels = (np.asarray(pil_image.convert('RGBA'),dtype=np.float32) * byte_to_normalized).ravel()
    return bpy_image

# Save the image to the current working directory
def save_image(image, name):
    img = pil_to_image(image)
    image.save(f"{name}.png")
    
def parse_response(response):
    for resp in response:
        for artifact in resp.artifacts:
            print('artifact type: '+ str(artifact.type))
            if artifact.type == 1:
                img = Image.open(io.BytesIO(artifact.binary))
                return img


# Create a new image from a prompt and save it to the current working directory
# if no name is provided, save the image the text of the prompt
def generate_texture_from_prompt(prompt,size,name):
    # if name is not defined, use the prompt as name
    if name is None:
        name = prompt
    # send request to stability api
    request = stability_api.generate(prompt, width=size[0], height=size[1])
    # parse response (PIL image is returned)
    image = parse_response(request)
    # save image to current working directory
    save_image(image, name)


    return request

generate_texture_from_prompt(prompt="A painting of luminescent translucent cast glass 3d xray abstract supernova portrait made of many vividly colored neon tube tribal masks filled with glowing uv blacklight led light trails and curved lightsaber glowing graffiti by okuda san miguel and kandinsky in a cubist face, in metallic light trail calligraphic light saber galaxies by okuda san miguel and kandinsky on a starry black canvas, galaxy gas brushstrokes, metallic flecked paint, metallic flecks, glittering metal paint, metallic paint, glossy flecks of iridescence, glow in the dark, uv, blacklight, uv blacklight, 8k, 4k, brush strokes, painting, highly detailed, iridescent texture, brushed metal", size=(512,512), name="DiffusionTexture_001")
