##
##
## Stablender Diffusion Load Script- For loading Stable Diffusion Web UI into blender
## This will work with a local installation or an instance of SD Web UI running in colab, all you need is the url to the
## gradio app.
## version: 0.0.0
## author: Shellworld (twitter: @shellworld1; github.com/shellward)
##
##


import os
import sys
import requests
from base64 import b64decode, b64encode
import io
from PIL import Image
import cv2
import numpy as np


# set vars
url = 'https://10247.gradio.app'
prompt="a painting of a triumphant battle on a lamp, an oil on canvas painting, cgsociety, fantastic realism, oil on canvas, academic art, matte painting"
cfg_scale=7.5
width=512,
height=512
steps=50
num_batches = 1
batch_size = 1

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

def predict(prompt, steps, cfg_scale, width, height, num_batches=1, batch_size=1):
    # get prediction
    response = requests.post(url + '/api/predict/', headers=headers, json={
    "fn_index": 4,
    "data": [
        prompt,
        steps,
        "k_lms",
        ["Normalize Prompt Weights (ensure sum of weights add up to 1.0)",
            "Save individual images",
            "Save grid",
            "Sort samples by prompt",
            "Write sample info files"],
        "RealESRGAN_x4plus",
        0,
        num_batches,
        batch_size,
        cfg_scale,
        "",
        width,
        height,
        None,
        0,
        ""
    ]
})
    return response.json()



def saveImage(image_data, filename):
    img = stringToRGB(image_data)
    img.save(filename)
    return img
    
def stringToRGB(base64_string):
    header, encoded = base64_string.split(",", 1)
    imgdata = b64decode(encoded)
    img = Image.open(io.BytesIO(imgdata))
    return img 

def RGBtoString(image):
    img = image_to_pil(image)
    img = image.convert('RGB')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    body = img_bytes.read()
    header= 'data:image/png;base64,'
    return header + b64encode(body).decode('utf-8')

def setCurrentObjectTextureToImage(image):
    # set current object texture to image
    bpy.context.object.active_material.active_texture.image = image

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

def image_to_pil(bpy_image):
    '''
    PIL image pixels is 2D array of byte tuple (when mode is 'RGB', 'RGBA') or byte (when mode is 'L')
    bpy image pixels is flat array of normalized values in RGBA order
    '''
    # setup PIL image reading
    width = bpy_image.size[0]
    height = bpy_image.size[1]
    bpy_pixels = bpy_image.pixels
    normalized_to_byte = 255.0
    num_pixels = width * height
    # setup bpy image
    channels = 4
    pil_image = Image.new('RGBA', (width, height))
    # bpy image has a flat RGBA array (similar to JS Canvas)
    pil_image.putdata(np.asarray(bpy_pixels).reshape(num_pixels, channels))
    return pil_image


def requestImage(prompt, steps, cfg_scale, width, height, fn):
    # get prediction
    response = predict(prompt, steps, cfg_scale, width, height)
    # save image
    print(response['data'])
    img = saveImage(response['data'][0][0], f"{fn}.png")
    #add image to bpy.data.images
    blender_image = pil_to_image(img)



def requestImages(prompt, steps, cfg_scale, width, height, num_batches, batch_size, fn):
    # get prediction
    response = predict(prompt, steps, cfg_scale, width, height, num_batches, batch_size)
    # save image
    print(response['data'])
    #skip the first response image, which is a collage
    for i, img in enumerate(response['data'][0]):
        if i == 0:
            continue
        img = saveImage(img, f"{fn}_{i}.png")
        blender_image = pil_to_image(img)





requestImages('rich watercolor painting of nature, trending on reddit, vintage', 50, 7.5, 512, 900,3,1, f"img_")
