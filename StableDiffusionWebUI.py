# Stablender Diffusion 0.0.1
# author: @shellworld
# license: MIT

# imports
import uuid
import bpy
import os
import sys
import requests
import json
from base64 import b64decode, b64encode
import io
from PIL import Image
import cv2
import numpy as np
from datetime import datetime
from enum import Enum

# constants
# function indices
class FunctionIndex(Enum):
    TEXT_TO_IMAGE = 4
    IMAGE_TO_IMAGE = 17
    IMAGE_TO_IMAGE_WITH_MASK = 16

# set vars
# comment out in operators for accepting input variables

# replace with gradio url
url = 'https://#####.gradio.app'
prompt = "landscape painting, oil on cavas, high detail 4k"
cfg_scale = 7.5
width = 512
height = 512
steps = 50
num_batches = 1
batch_size = 1
strength = .71

# set headers for requests to StableDiffusion Web Ui
headers = {
    "authority": f"{url}",
    "method": "POST",
    "path": "/api/predict/",
    "scheme": "https",
    "accept": "*/*",
    "accept-encoding": "gzip, deflate, br",
    "accept-language": "en-US,en;q=0.9",
    "dnt": "1",
    "origin": f"{url}",
    "referer": f"{url}",
    "sec-ch-ua": "`\"Chromium`\";v=\"104\", `\" Not A;Brand`\";v=\"99\", `\"Google Chrome`\";v=\"104\"\"",
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": "`\"Windows`\"",
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin"
}

# constant static variables
MAX_STEPS = 400
MIN_STEPS = 10
MAX_PROMPT_LENGTH = 1000
CFG_SCALE_FLOOR = 0.1
CFG_SCALE_CEILING = 10.0
SIZE_MIN = 512
SIZE_MAX = 768
HIGH_BATCH_NUMBER = 16
HIGH_BATCH_NUMBER_ENABLED = False
MAX_LENGTH_FILE_NAME=24


# predict
def predict(prompt:str, steps:int, cfg_scale:float, width:int, height:int, num_batches=1, batch_size=1, image_string="", mask_string="", function_index=FunctionIndex.TEXT_TO_IMAGE,strength=.50):
    # get prediction

    # set function index based on input
    # this will grab the corresponding enum value
    fn_index = function_index.value


    # define validation functions

    def checkPrompt(prompt):
        #prompt is not null or empty
        if prompt is None or prompt == "":
            raise (Exception("prompt is required"))
        # prompt is a valid string
        if type(prompt) != str:
            raise (Exception("prompt must be a string"))
        # prompt is not too long
        if len(prompt) > MAX_PROMPT_LENGTH:
            raise (
                Exception(f"prompt must be less than {str(MAX_PROMPT_LENGTH)} characters"))

    def checkSteps(steps):
        if steps is None:
            raise (Exception("steps is required"))
        if type(steps) != int:
            raise (Exception("steps must be an integer"))
        if steps < MIN_STEPS:
            raise (Exception(f"steps must be greater than {str(MIN_STEPS)}"))
        if steps > MAX_STEPS:
            #steps is not null or empty
            if steps is None or steps == "":
                raise (Exception("steps are required"))
            # steps is a valid integer
            if type(steps) != int:
                raise (Exception("steps must be an integer"))
            # steps are within the range of 10 to 400
            if steps < MIN_STEPS or steps > MAX_STEPS:
                raise (
                    Exception(f"steps must be between {str(MIN_STEPS)} and {str(MAX_STEPS)}"))

    def checkCfgScale(cfg_scale):
        #cfg_scale is not null or empty
        if cfg_scale is None or cfg_scale == "":
            raise (Exception("cfg_scale is required"))
        # cfg_scale is a valid float
        if type(cfg_scale) != float:
            raise (Exception("cfg_scale must be a float"))
        # cfg_scale is within the range of -CFG_SCALE_FLOOR to CFG_SCALE_CEILING
        if cfg_scale < CFG_SCALE_FLOOR or cfg_scale > CFG_SCALE_CEILING:
            raise (Exception(
                f"cfg_scale must be between {str(CFG_SCALE_FLOOR)} and {str(CFG_SCALE_CEILING)}"))

    def checkWidth(width):
        #width is not null or empty
        if width is None or width == "":
            raise (Exception("width is required"))
        # width is a valid integer
        if type(width) != int:
            raise (Exception("width must be an integer"))
        # width is greater than SIZE_MIN and less than SIZE_MAX
        if width < SIZE_MIN or width > SIZE_MAX:
            raise (
                Exception(f"width must be between {str(SIZE_MIN)}px and {str(SIZE_MAX)}px"))

    def checkHeight(height):
        #height is not null or empty
        if height is None or height == "":
            raise (Exception("height is required"))
        # height is a valid integer
        if type(height) != int:
            raise (Exception("height must be an integer"))
        # height is greater than SIZE_MIN and less than SIZE_MAX
        if height < SIZE_MIN or height > SIZE_MAX:
            raise (
                Exception(f"height must be between {str(SIZE_MIN)}px and {str(SIZE_MAX)}px"))

    def checkNumBatches(num_batches):
        #num_batches is not null or empty
        if num_batches is None or num_batches == "":
            raise (Exception("num_batches is required"))
        # num_batches is a valid integer greater than 0
        if type(num_batches) != int or num_batches <= 0:
            raise (Exception("num_batches must be an integer greater than zero"))
        # num_batches is greater than 16
        if num_batches > HIGH_BATCH_NUMBER and not HIGH_BATCH_NUMBER_ENABLED:
            raise (Exception(
                f"num_batches must be less than {str(HIGH_BATCH_NUMBER)} or set HIGH_BATCH_NUMBER_ENABLED to True"))

    # Developer Note:
    # It might make sense to disable the batch size option
    # in your interface.

    def checkBatchSize(batch_size):
        if batch_size is None or batch_size == "":
            raise (Exception("batch_size is required"))
        # batch_size is a valid integer greater than 0
        if type(batch_size) != int or batch_size <= 0:
            raise (Exception("batch_size must be an integer greater than zero"))
        # batch_size is greater than 1
        if batch_size > 1:
            raise (Exception(f"batch_size must be set to one for now."))

    def checkImageString(image_string):
        if image_string is not None or image_string != "":
            # image_string is a valid string
            if type(image_string) != str:
                print(image_string)
                raise (Exception(f"image_string must be a string, but it is currently {type(image_string)}"))
                # image string is a valid base64 png datastream
            if not image_string.startswith("data:image/png;base64,"):
                print(f"DEBUG:{image_string}")
                raise (Exception("image_string must be a valid base64 png datastream"))
                

    def checkMaskString(mask_string):
        if mask_string is not None or mask_string != "":
            # mask_string is a valid string
            if type(mask_string) != str:
                raise (Exception("mask_string must be a string"))
                # mask string is a valid base64 png datastream
            if not mask_string.startswith("data:image/png;base64,"):
                raise (Exception("mask_string must be a valid base64 png datastream"))

    def checkStrength(strength):
        if strength is not None or strength != "":
            # strength is a valid float
            if type(strength) != float:
                raise (Exception("strength must be a float"))
            # strength is within the range of 0 to 1
            if strength < 0 or strength > 1:
                raise (Exception(f"strength must be between 0 and 1"))

    def validate(prompt, steps, cfg_scale, width, height, num_batches, batch_size, image_string, mask_string, strength):
        checkPrompt(prompt)
        checkSteps(steps)
        checkCfgScale(cfg_scale)
        checkWidth(width)
        checkHeight(height)
        checkNumBatches(num_batches)
        checkBatchSize(batch_size)
        if function_index == FunctionIndex.IMAGE_TO_IMAGE or FunctionIndex.IMAGE_TO_IMAGE_WITH_MASK:
            checkImageString(image_string)
            checkStrength(strength)
            if function_index== FunctionIndex.IMAGE_TO_IMAGE_WITH_MASK:
                checkMaskString(mask_string)

    def validateResponse(response):
        if response is None or response == "":
            raise (Exception(f"Something went wrong: {response}"))
        if response.status_code != 200:
            raise (Exception(f"Something went wrong: {response}"))

    # validate the input
    validate(prompt, steps, cfg_scale, width, height, num_batches,
             batch_size, image_string, mask_string, strength)

    # depending on which mode we are in, we need to set the data differently

    # txt2img
    if function_index == FunctionIndex.TEXT_TO_IMAGE:
        data = [prompt, steps, "k_lms",
                ["Normalize Prompt Weights (ensure sum of weights add up to 1.0)",
                 "Save individual images",
                 "Save grid",
                 "Sort samples by prompt",
                 "Write sample info files"],
                "RealESRGAN_x4plus", 0, num_batches, batch_size,
                cfg_scale, "", width, height, None, 0, ""
                ]

    # img2img
    elif function_index == FunctionIndex.IMAGE_TO_IMAGE:
        data = [prompt, "Crop", image_string, "Keep masked area",
                3, steps, "k_lms",
                ["Normalize Prompt Weights (ensure sum of weights add up to 1.0)",
                 "Save individual images",
                 "Save grid",
                 "Sort samples by prompt",
                 "Write sample info files"],
                "RealESRGAN_x4plus",
                num_batches, batch_size, cfg_scale, strength,
                None, width, height, "Just resize", None
                ]

    # img2img with mask
    elif function_index == FunctionIndex.IMAGE_TO_IMAGE_WITH_MASK:
        data = [prompt, "Mask", {"image": image_string, "mask": mask_string},
                "Keep masked area", 3, steps, "k_lms",
                ["Normalize Prompt Weights (ensure sum of weights add up to 1.0)",
                 "Save individual images",
                 "Save grid",
                 "Sort samples by prompt",
                 "Write sample info files"],
                "RealESRGAN_x4plus",
                num_batches, batch_size, cfg_scale, strength,
                None, width, height, "Just resize", None
                ]

    response = requests.post(url + '/api/predict/', headers=headers, json={
    "fn_index": fn_index,
    "data":data
    })

    # validate the response
    validateResponse(response)
    
    return response.json()




# utility classes

# function that converts png datastream to Image
def stringToRGB(base64_string:str):
    header, encoded = base64_string.split(",", 1)
    imgdata = b64decode(encoded)
    img = Image.open(io.BytesIO(imgdata))
    return img

# function that converts Image to png datastream
def rgbToString(image:Image.Image):
    img = image.pixels
    img = Image.fromarray(img.reshape(image.size[1], image.size[0], 4))
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    img_string = b64encode(img_bytes.read()).decode('ascii')
    return img_string


def parseResults(response):
#return array of b64 strings from response[data][0]

    # get the data property's zeroth index
    data = response["data"][0]

    #if the length of data is zero, raise an exception
    if len(data) == 0:
        raise (Exception("No data returned"))
    
    elif len(data) == 1:
        # if the length of data is one, then we have a single image
        # return the image
        return [data[0]]

    else:
        images = []
        for i in range(len(data)):
            # if data>1, the first image will be a grid, ignore this image
            if i == 0:
                continue
            # return multiple images
            else:
                images.append(data[i])
        return images

def convertPNGDatastreamsToBPYImages(base64_strings:str):
    images = []
    #convert to CV2 images
    for base64_string in base64_strings:
        images.append(stringToRGB(base64_string))
    #now convert the cv2 images to bpy images
    for i in range(len(images)):
       
        images[i] = bpy.data.images.new(f"image_{i}", images[i].size[0], images[i].size[1], alpha=True)
        images[i].pixels = images[i].pixels[:]
        images[i].filepath_raw = f"image_{i}.png"
        images[i].file_format = 'PNG'
    return images

def convertBPYImageToBase64PNGDataStream(referenceToBPYImage):
    selectedImage = bpy.data.images[referenceToBPYImage]

    #convert the bpy image to a cv2 image
    img = cv2.imread(f"{bpy.path.abspath(selectedImage.filepath_raw)}")
    #convert the cv2 image to a b64 string
    _, im_arr = cv2.imencode('.jpg', img)  # im_arr: image in Numpy one-dim array format.
    im_bytes = im_arr.tobytes()
    im_b64 = b64encode(im_bytes).decode('ascii')
    im_b64 = "data:image/png;base64," + im_b64
    return im_b64
    

def pil_to_image(pil_image, name='NewImage'):
    '''
    PIL image pixels is 2D array of byte tuple (when mode is 'RGB', 'RGBA') or byte (when mode is 'L')
    bpy image pixels is flat array of normalized values in RGBA order
    '''
    # setup PIL image reading

    # swap red and blue channels
    pil_image = pil_image.transpose(Image.FLIP_TOP_BOTTOM)
    # convert to bpy image
    

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

#save bpy images     
def saveImage(image_data, filename):
    img = stringToRGB(image_data)
    img.save(filename)
    return img

# generate a safe filename
def generateSafeNameFromPromptAndIndex(prompt, index):
    prompt = prompt.replace(" ", "_")
    prompt = prompt.replace(",", "-")
    prompt = prompt[0:MAX_LENGTH_FILE_NAME]
    return prompt + "_" + str(index)

## MAIN FUNCTIONS

# function that converts base64 png datastream to Image
def requestImg(prompt, steps, cfg_scale, width, height, fn, bpy_image=None, mask_image=None, batch_num=1, batch_size=1,  strength=50):
    #request the image
    response = predict(prompt, steps, cfg_scale, width, height,
                       num_batches, batch_size,bpy_image, mask_image, fn)

    #results will be a list of base64 strings
    results = parseResults(response)
    img = saveImage(results[0], f"{generateSafeNameFromPromptAndIndex(prompt, 0)}.png")
    #add image to bpy.data.images
    blender_image = pil_to_image(img, f"{generateSafeNameFromPromptAndIndex(prompt, 0)}")
    return f"{generateSafeNameFromPromptAndIndex(prompt, 0)}"


#Text to image example
#requestImg(prompt, steps, cfg_scale, width, height, FunctionIndex.TEXT_TO_IMAGE)

#Image to image example
#requestImg(prompt, steps, cfg_scale, width, height, FunctionIndex.IMAGE_TO_IMAGE, bpy_image=convertBPYImageToBase64PNGDataStream("V"), strength=strength)

