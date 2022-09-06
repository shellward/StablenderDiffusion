##
##
## Stablender Diffusion Load Script- For loading Stable Diffusion Web UI into blender
## This will work with a local installation or an instance of SD Web UI running in colab, all you need is the url to the
## gradio app.
## version: 0.0.1
## author: Shellworld (twitter: @shellworld1; github.com/shellward)
##
##


#imports
import uuid
import bpy
import os
import sys
import requests
from base64 import b64decode, b64encode
import io
from PIL import Image
import cv2
import numpy as np
from datetime import datetime

# set vars

url = 'https://13052.gradio.app'
prompt=" moonlight cube, computer graphics by Ryūsei Kishida,, zbrush central, generative art, made of flowers, cosmic horror, made of vines"
cfg_scale=7.5
width=512
height=512
steps=50
num_batches = 2
batch_size = 1

# set up count for number of runs
if 'count' in locals():
    count=count+1
else:
    count=0

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

# There are two separate prediction functions, but as these change with 
# the stable diffusion web ui, I'm keeping these separate until an API
# /docs are available.

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

def predictFromImage(img_string, strength, prompt, steps, cfg_scale, width, height,num_batches=1, batch_size=1):
    response = requests.post(url + '/api/predict/', headers=headers, json={
        "fn_index": 17,
        "data": [
            prompt,
            "Crop",
            img_string,
            "Keep masked area",
            3,
            steps,
            "k_lms",
            ["Normalize Prompt Weights (ensure sum of weights add up to 1.0)",
                "Save individual images",
                "Save grid",
                "Sort samples by prompt",
                "Write sample info files"],
            "RealESRGAN_x4plus",
            num_batches,
            batch_size,
            cfg_scale,
            strength,
            None,
            width,
            height,
            "Just resize",
            None
        ]
    })
    return response.json()

def predictFromImageWithMask(img_string, mask_string, strength, prompt, steps, cfg_scale, width, height, num_batches=1, batch_size=1):
    response = requests.post(url + '/api/predict/', headers=headers, json={
        "fn_index": 16,
        "data": [
            prompt,
            "Mask",
            {"image": img_string,
            "mask": mask_string},
            "Keep masked area",
            3,
            steps,
            "k_lms",
            ["Normalize Prompt Weights (ensure sum of weights add up to 1.0)",
                "Save individual images",
                "Save grid",
                "Sort samples by prompt",
                "Write sample info files"],
            "RealESRGAN_x4plus",
            num_batches,
            batch_size,
            cfg_scale,
            strength,
            None,
            width,
            height,
            "Just resize",
            None
        ]
    })
    return response.json()

# save the image to the local directory
def saveImage(image_data, filename):
    img = stringToRGB(image_data)
    img.save(filename)
    return img


# convert datastring to PIL image
def stringToRGB(base64_string):
    header, encoded = base64_string.split(",", 1)
    imgdata = b64decode(encoded)
    img = Image.open(io.BytesIO(imgdata))
    return img 

# convert bpy image to datastring
def rgbToString(image):
    img = image.pixels
    img = Image.fromarray(img.reshape(image.size[1], image.size[0], 4))
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    img_string = b64encode(img_bytes.read()).decode('ascii')
    return img_string
    

#trying a replacement 
# def rgb2string2(image):
    # img = image_to_pil(image)
    # img = image.convert('RGB')
    # img_bytes = io.BytesIO()
    # img.save(img_bytes, format='PNG')
    # img_bytes.seek(0)
    # body = img_bytes.read()
    # header= 'data:image/png;base64,'
    # return header + b64encode(body).decode('utf-8')


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


def image_to_pil(image):
    '''
    bpy image pixels is flat array of normalized values in RGBA order
    PIL image pixels is 2D array of byte tuple (when mode is 'RGB', 'RGBA') or byte (when mode is 'L')
    '''
    # setup bpy image
    width = image.size[0]
    height = image.size[1]
    bpy_pixels = image.pixels
    # setup PIL image
    channels = 4
    mode = 'RGBA'
    pil_image = Image.new(mode, (width, height))
    # PIL image has a flat RGBA array (similar to JS Canvas)
    pil_image.putdata(bpy_pixels, scale=1.0, offset=0.0)
    return pil_image




def requestImage(prompt, steps, cfg_scale, width, height, fn):
    # get prediction
    response = predict(prompt, steps, cfg_scale, width, height)
    # save image
    print(response['data'])
    img = saveImage(response['data'][0][0], f"{fn}.png")
    #add image to bpy.data.images
    blender_image = pil_to_image(img, f"{fn}")
    return f"{fn}"

def requestImages(prompt, steps, cfg_scale, width, height, fn, num_batches, batch_size):
    img_list = []
    response = predict(prompt, steps, cfg_scale, width, height, num_batches, batch_size)
    for i in range(num_batches):
    # save image
        print(response['data'])
        for idx, image in enumerate(response['data'][0]):
            if idx == 0:
                pass
            else:
                img = saveImage(image, f"{fn}_{idx}.png")
                #add image to bpy.data.images
                blender_image = pil_to_image(img, f"{fn}_{idx}")
                img_list.append(f"{fn}_{idx}")
    return img_list


                

def requestImageToImage(bpy_image, prompt, steps, cfg_scale, width, height, fn):
   #convert bpy image to PIL image
   pil_image = image_to_pil(bpy_image)
   image_string = rgbToString(pil_image)
   response = predictFromImage(image_string, prompt, steps, cfg_scale, width, height)
   # save image
   print(response['data'])
   img = saveImage(response['data'][0][0], f"{fn}.png")
   #add image to bpy.data.images
   blender_image = pil_to_image(img)
   return pil_to_image(img)

def requestImageToImageWithMask(bpy_image, mask_image, prompt, steps, cfg_scale, width, height, fn):
    #convert bpy image to PIL image
    pil_image = image_to_pil(bpy_image)
    pil_mask = image_to_pil(mask_image)
    image_string = rgbToString(pil_image)
    mask_string = rgbToString(pil_mask)
    response = predictFromImageWithMask(image_string, mask_string, prompt, steps, cfg_scale, width, height)
    # save image
    print(response['data'])
    img = saveImage(response['data'][0][0], f"{fn}.png")
    #add image to bpy.data.images
    blender_image = pil_to_image(img)
    return pil_to_image(img)




def requestImageToImages(bpy_image, prompt, steps, cfg_scale, width, height, fn, num_batches, batch_size):
   #convert bpy image to PIL image
   pil_image = image_to_pil(bpy_image)
   image_string = rgbToString(pil_image)
   response = predictFromImage(image_string, prompt, steps, cfg_scale, width, height, num_batches, batch_size)
   # save image
   print(response['data'])
   for idx, image in enumerate(response['data'][0]):
       if idx == 0:
           pass
       else:
           img = saveImage(image, f"{fn}_{idx}.png")
           #add image to bpy.data.images
           blender_image = pil_to_image(img)

def requestImageToImagesWithMask(bpy_image, mask_image, prompt, steps, cfg_scale, width, height, fn, num_batches, batch_size):
    #convert bpy image to PIL image
    pil_image = image_to_pil(bpy_image)
    pil_mask = image_to_pil(mask_image)
    image_string = rgbToString(pil_image)
    mask_string = rgbToString(pil_mask)
    response = predictFromImageWithMask(image_string, mask_string, prompt, steps, cfg_scale, width, height, num_batches, batch_size)
    # save image
    print(response['data'])
    for idx, image in enumerate(response['data'][0]):
        if idx == 0:
            pass
        else:
            img = saveImage(image, f"{fn}_{idx}.png")
            #add image to bpy.data.images
            blender_image = pil_to_image(img)

def uuidNameGenerator():
    return str(uuid.uuid4())

def createNewMaterialWithImageAsBaseColor(image):

    #create new image texture with image
    unique_name_tex = uuidNameGenerator()
    bpy.data.textures.new(unique_name_tex, 'IMAGE')
    bpy.data.textures[unique_name_tex].image = bpy.data.images[image]

    #create new material
    unique_name = uuidNameGenerator()
    bpy.data.materials.new(unique_name)
    bpy.data.materials[unique_name].use_nodes = True
    bpy.data.materials[unique_name].node_tree.nodes.clear()


    #create material output node
    output_node = bpy.data.materials[unique_name].node_tree.nodes.new('ShaderNodeOutputMaterial')
    output_node.location = (20,0)
    
    #connnect image texture node to material output 
    image_node = bpy.data.materials[unique_name].node_tree.nodes.new('ShaderNodeTexImage')
    image_node.image = bpy.data.textures[unique_name_tex].image
    image_node.location = (0,0)
    bpy.data.materials[unique_name].node_tree.links.new(image_node.outputs[0], bpy.data.materials[unique_name].node_tree.nodes['Material Output'].inputs[0])

    #set material to object
    bpy.context.object.active_material = bpy.data.materials[unique_name]

    #fake a user so the material is saved
    bpy.data.materials[unique_name].use_fake_user = True

def createNewBSDFMaterialWithImageAsBaseColorAndNormalHeight(image):

    #create new image texture with image
    unique_name_tex = uuidNameGenerator()
    bpy.data.textures.new(unique_name_tex, 'IMAGE')
    bpy.data.textures[unique_name_tex].image = bpy.data.images[image]

    #create new material
    unique_name = uuidNameGenerator()
    bpy.data.materials.new(unique_name)
    bpy.data.materials[unique_name].use_nodes = True
    bpy.data.materials[unique_name].node_tree.nodes.clear()


    #create material output node
    output_node = bpy.data.materials[unique_name].node_tree.nodes.new('ShaderNodeOutputMaterial')
    output_node.location = (20,0)
    
    #connnect image texture node to material output 
    image_node = bpy.data.materials[unique_name].node_tree.nodes.new('ShaderNodeTexImage')
    image_node.image = bpy.data.textures[unique_name_tex].image
    image_node.location = (0,0)
    bpy.data.materials[unique_name].node_tree.links.new(image_node.outputs[0], bpy.data.materials[unique_name].node_tree.nodes['Material Output'].inputs[0])

    #create normal height node
    normal_height_node = bpy.data.materials[unique_name].node_tree.nodes.new('ShaderNodeNormalMap')
    normal_height_node.location = (0,200)
    bpy.data.materials[unique_name].node_tree.links.new(image_node.outputs[0], normal_height_node.inputs[0])

    #set material to object
    bpy.context.object.active_material = bpy.data.materials[unique_name]

    #fake a user so the material is saved
    bpy.data.materials[unique_name].use_fake_user = True

def createNewMaterialsWithImageAsABaseColor(image_list, bsdf = False):
    if bsdf:
        for image in image_list:
            createNewBSDFMaterialWithImageAsBaseColorAndNormalHeight(image)
    else:
        for image in image_list:
            createNewMaterialWithImageAsBaseColor(image)

def setMaterialToActiveObject(material):
    bpy.context.object.active_material = bpy.data.materials[material]

def clearImagesAndMaterials():
    for image in bpy.data.images:
        bpy.data.images.remove(image)
    for material in bpy.data.materials:
        bpy.data.materials.remove(material)

def makeMask(img):
    #create a mask from selecting the 0th index pixel in a bpy.image and getting all pixels !=0th idx color? 
    #create new image texture with image
    unique_name_tex = uuidNameGenerator()
    bpy.data.textures.new(unique_name_tex, 'IMAGE')
    bpy.data.textures[unique_name_tex].image = bpy.data.images[img]

    #select 0th pixel index's color
    bpy.data.textures[unique_name_tex].image.pixels[0][0]

    #create new image mask where pixels != 0th pixel index color
    unique_name_mask = uuidNameGenerator()
    bpy.data.images.new(unique_name_mask, width=bpy.data.textures[unique_name_tex].image.size[0], height=bpy.data.textures[unique_name_tex].image.size[1], alpha=True)
    bpy.data.images[unique_name_mask].pixels[:] = bpy.data.textures[unique_name_tex].image.pixels[:]
    bpy.data.images[unique_name_mask].pixels[:] = [x != bpy.data.textures[unique_name_tex].image.pixels[0][0] for x in bpy.data.images[unique_name_mask].pixels]

    #create new image mask node
    mask_node = bpy.data.materials[unique_name].node_tree.nodes.new('ShaderNodeTexImage')
    mask_node.image = bpy.data.images[unique_name_mask]
    mask_node.location = (0,0)
    bpy.data.materials[unique_name].node_tree.links.new(mask_node.outputs[0], bpy.data.materials[unique_name].node_tree.nodes['Material Output'].inputs[0])

    #create new image from image mask
    unique_name_mask_img = uuidNameGenerator()
    bpy.data.images.new(unique_name_mask_img, width=bpy.data.textures[unique_name_tex].image.size[0], height=bpy.data.textures[unique_name_tex].image.size[1], alpha=True)
    bpy.data.images[unique_name_mask_img].pixels[:] = bpy.data.images[unique_name_mask].pixels[:]
    
    return bpy.data.images[unique_name_mask_img]

    

dt = datetime.today()  # Get timezone naive now
seconds = dt.timestamp()
req_img = requestImage(prompt, steps, cfg_scale, width, height, f"{seconds}")
createNewMaterialWithImageAsBaseColor(req_img)










