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
#import dlib

dlib_fun=opf.AlignDlib('./lib/dlib_model/shape_predictor_68_face_landmarks.dat')
my_hr_style={
                #'border-width' : '4px',
                'border-top': '2px solid #21ABCD',
                'margin-top': '0.5rem',
                'margin-bottom': '0.5rem',
                'margin-left': 'auto',
                'margin-right': 'auto'
            }
h5_style = {
            'textAlign': 'center',
            'color': 'black',
            'margin-top': '1.0rem',
            'margin-bottom': '1.0rem'
        }
img_style = { 'margin' : 'auto',
              #'align' : 'middle',
              'display': 'block',
            #   'width':'80%',
            #   'height':'80%',
            #   'textAlign': 'center',
            #   'vertical-align': 'middle',
            #   'align':'right',
            #   'textAlign':'center'
            'margin-top': '3.0rem',
            'margin-bottom': '1.0rem'
            }
bar_style = { 'margin' : 'auto',
              'display': 'inline-block',
              'width':'100%',
              'height':'100%',
              'textAlign': 'center',
              'vertical-align': 'middle',
              'align':'left'}
app = dash.Dash()

app.scripts.config.serve_locally = True

app.layout = html.Div([
    html.H2(
        children='Facial Expression Recognition',
        style={
            'textAlign': 'center',
            'color': 'black',
            'margin-top': '1.0rem',
            'margin-bottom': '1.0rem'
        }
    ),
    #html.Hr(),
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A(' Select A Image')
        ]),
        style={
            'width': '40%',
            'height': '40px',
            'lineHeight': '40px',
            'borderWidth': '3px',
            'borderStyle': 'solid',
            'borderRadius': '50px',
            'borderColor': '#007FFF',
            #'backgroundColor':'rgb(242, 242, 242)',
            'textAlign': 'center',

            #'marginLeft':'10px',

            'margin': 'auto'
            #'padding': '10px 100px 10px 200px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Hr(
            style=my_hr_style
    ),
    html.Div(id='output-image-upload'),
])

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

def layout_plot(img):
    #img = arrtobase64(img)
    img = rescale(img,8,anti_aliasing=False)
    img = 'data:image/png;base64,{}'.format(encode(img))
    result = html.Div([
                    html.Div([
                          html.Div(
                                    html.Img(src=img, style=img_style),
                                    className="six columns"),
                          html.Div(
                                    dcc.Graph(id='prediction-bar-graph',
                                              figure={
                                                        'data': [
                                                                 {'x': ['Angry'+'\U0001f620',
                                                                        'Fear'+'\U0001f628',
                                                                        'Happy'+'\U0001f604',
                                                                        'Sad'+'\U0001f622',
                                                                        'Surprise'+'\U0001f632',
                                                                        'Neutral'+u'\U0001f610'],
                                                                  'y': [0.0  , 0.0  , 17.0, 0.0  , 0.0  , 83.0],
                                                                  'type': 'bar'}
                                                                 ],
                                                         'layout': { 'height':'50%',
                                                                     'margin':{'l': 60, 'b': 60, 't': 0, 'r': 0},
                                                                     'font': {'size' : 20},
                                                                     'xaxis':{'title': 'Facial Emotion'},
                                                                     'yaxis':{'title': 'Probability'}
                                                        #             'title': 'Dash Data Visualization'
                                                                   }
                                                    }  , style={'fontWeight':300}
                                                ),
                                    className="six columns"),
                          ],className="row"),
                html.Hr(style=my_hr_style)
    ])
    return result

def side_by_side_plot(list_images):
    total_layout = []
    for img in list_images:
        total_layout.append(layout_plot(img))

    return total_layout

        #
def parse_contents(contents, filename):

    aligned_faces = get_face(contents)
    #encoded_imges = [ arrtobase64(img) for img in aligned_faces]


    images = [ html.Div([
            html.H5(filename, style= h5_style),
        #html.H6(datetime.datetime.fromtimestamp(date)),
        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
            html.Img(src=contents, style = { 'margin' : 'auto', 'display': 'block'}),
            html.Hr(style=my_hr_style),
            ])
    ]
    #images.extend(layout_plot(contents))
    images.extend(side_by_side_plot(aligned_faces))
#    import pdb;pdb.set_trace()
    return html.Div(images)

@app.callback(Output('output-image-upload', 'children'),
              [Input('upload-image', 'contents'),
               Input('upload-image', 'filename')])
               #Input('upload-image', 'last_modified')])
def update_output(list_of_contents, list_of_names):
    if list_of_contents is not None:
        children = [ parse_contents(c, n) for c, n in
            zip(list_of_contents, list_of_names)]
        return children


app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})

if __name__ == '__main__':
    app.run_server(debug=True, port=8819)
