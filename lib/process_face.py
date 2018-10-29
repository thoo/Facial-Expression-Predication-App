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
