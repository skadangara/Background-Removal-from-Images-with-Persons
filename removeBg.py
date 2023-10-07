import cv2
from urllib.request import urlretrieve
import torch
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt
import numpy as np
import urllib as urllib
import os

from flask import Flask, request
from flask import send_file

app = Flask(__name__)

@app.route('/',methods=['GET'])
def index():
    home = "This is a web service for background removal from images. If the image is in remote location \
             use: http://127.0.0.1:5000/remove_img_background , If the image is in local path \
             use: http://127.0.0.1:5000/rm_img_bg_local"
    return home

def load_model():
    model = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=True)
    model.eval()
    return model

def make_transparent_foreground(pic, mask):
    # split the image into channels
    b, g, r = cv2.split(np.array(pic).astype('uint8'))
    # add an alpha channel with and fill all with transparent pixels (max 255)
    a = np.ones(mask.shape, dtype='uint8') * 255
    # merge the alpha channel back
    alpha_im = cv2.merge([b, g, r, a], 4)
    # create a transparent background
    bg = np.zeros(alpha_im.shape)
    # setup the new mask
    new_mask = np.stack([mask, mask, mask, mask], axis=2)
    # copy only the foreground color pixels from the original image where mask is set
    foreground = np.where(new_mask, alpha_im, bg).astype(np.uint8)

    return foreground
       
def remove_background(model, input_file):
    input_image = Image.open(input_file)
    preprocess = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available : this block can be enabled if CUDA is available.
#     if torch.cuda.is_available():
#         input_batch = input_batch.to('cuda')
#     model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)

    # create a binary (black and white) mask of the profile foreground
    mask = output_predictions.byte().cpu().numpy()
    background = np.zeros(mask.shape)
    bin_mask = np.where(mask, 255, background).astype(np.uint8)

    foreground = make_transparent_foreground(input_image ,bin_mask)

    return foreground, bin_mask

## Function for background removal of images present in remote location through URL.       
@app.route('/remove_img_background',methods=['POST'])
def api_rmbg():
    
    # Download the image
    data = request.get_json()
    url = data['url']
    try:
        url_resp = urllib.request.urlopen(url)
    except urllib.error.HTTPError as e:
        #body = e.readlines()
        return e.reason
    except urllib.error.URLError as e:
        return e.reason
    # Check the response is success
    
    if(url_resp.code == 200):
       # save image to local path
        name = "raw_image"
        fullname = str(name)+".jpg"
        urllib.request.urlretrieve(url,fullname)
    else:
        return "The image does not present in the location or permission denied"    
    
    #call deeplab model
    deeplab_model = load_model()
    foreground, bin_mask = remove_background(deeplab_model, fullname)
    
    #Save the foreground image locally
    converted_img = "bg_removed_img.png"
    Image.fromarray(foreground).save(converted_img)
    
    # Return the new image with background removed
    return send_file(converted_img, mimetype='image/png')

## Function for background removal of images present in the local path.
@app.route('/rm_img_bg_local',methods=['POST'])
def api_rmbg_local():
    
    data = request.get_json()
    image_path = data['image_path']

    # Check image exists in local path
    if not (os.path.exists(image_path)):        
        return "The image does not present in the location"    
    else:            

        #call deeplab model
        deeplab_model = load_model()
        foreground, bin_mask = remove_background(deeplab_model, image_path)

        #Save the foreground image locally
        converted_img = "bg_removed_img.png"
        Image.fromarray(foreground).save(converted_img)

        # Return the new image with background removed
        return send_file(converted_img, mimetype='image/png')
       
@app.route('/remove_background_save',methods=['POST'])
def api_rmbg_save():
    
    data = request.get_json()
    url = data['url']
    name = "raw_image"
    fullname = str(name)+".jpg"
    # save image to local path
    urllib.request.urlretrieve(url,fullname)
    
    #call deeplab model
    deeplab_model = load_model()
    foreground, bin_mask = remove_background(deeplab_model, fullname)
    # return the image after background removal, which will use for saving the image in remote location.
    converted_img = "bg_removed_img.png"
    Image.fromarray(foreground).save(converted_img)



if __name__ == "__main__":
    app.run()