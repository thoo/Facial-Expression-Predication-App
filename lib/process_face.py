import base64
import datetime
import io
import pdb

import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import openface as opf
import skimage
from dash.dependencies import Input, Output
from imageio import imread
from PIL import Image
from skimage.color import rgb2gray
from skimage.transform import rescale


dlib_fun=opf.AlignDlib('./lib/dlib_model/shape_predictor_68_face_landmarks.dat')
def _get_face(contents):
    if contents.startswith('data:image/jpeg;base64,'):
        contents=contents[len('data:image/jpeg;base64,'):]
    imgdata = base64.b64decode(contents)
    arr = np.frombuffer(imgdata,np.uint8)
    img = imread(io.BytesIO(base64.b64decode(contents)))
    #arr = io.imread(imgdata,plugin='imageio')
    # arr = base64.decodestring(contents.encode('ascii'))
    # arr = np.frombuffer(arr, dtype = np.float)
    #pdb.set_trace()
    return img

def arrtobase64(arr):
    return 'data:image/png;base64,{}'.format(base64.encodestring(arr).decode('utf-8'))

def encode(image) -> str:

    # convert image to bytes
    with io.BytesIO() as output_bytes:
        PIL_image = Image.fromarray(skimage.img_as_ubyte(image))
        PIL_image.save(output_bytes, 'JPEG') # Note JPG is not a vaild type here
        bytes_data = output_bytes.getvalue()

    # encode bytes to base64 string
    base64_str = str(base64.b64encode(bytes_data), 'utf-8')
    return base64_str

def get_face(contents):
    img = _get_face(contents)
    dets2=dlib_fun.getAllFaceBoundingBoxes(img)

    image_list=[]
    aligned_image_list=[]
    pad_v = int(np.shape(img)[0]/len(dets2)*0.2)# img.shape()[0]/len(dets2)*0.1
    pad_h = int(np.shape(img)[1]/len(dets2)*0.2)
    pad = min(pad_v,pad_h)
    for d in dets2:
        image_list.append(img[d.top()-pad:d.bottom()+pad, d.left()-pad:d.right()+pad])
        align2=dlib_fun.align(48,img,bb=d,landmarkIndices=opf.AlignDlib.INNER_EYES_AND_BOTTOM_LIP)
        align2=rgb2gray(align2)
        aligned_image_list.append(align2)
#    pdb.set_trace()
    return aligned_image_list
