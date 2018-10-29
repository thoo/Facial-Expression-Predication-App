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

from lib.process_face import *
from lib.style import *
from lib.tensorflow_model import *

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


def layout_plot(num, img, predictions):
    #img = arrtobase64(img)
    img = rescale(img,8,anti_aliasing=False)
    img = 'data:image/png;base64,{}'.format(encode(img))
    result = html.Div([
                    html.Div([
                          html.Div(
                                    html.Img(src=img, style=img_style),
                                    className="six columns"),
                          html.Div(
                                    dcc.Graph(id=str(num),figure={
                                                        'data': [
                                                                 {'x': ['Angry'+'\U0001f620',
                                                                        'Fear'+'\U0001f628',
                                                                        'Happy'+'\U0001f604',
                                                                        'Sad'+'\U0001f622',
                                                                        'Surprise'+'\U0001f632',
                                                                        'Neutral'+u'\U0001f610'],
                                                                  'y': predictions,
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

def side_by_side_plot(list_images,y_prediction):
    total_layout = []
    for num,(img,predictions) in enumerate(zip(list_images,y_prediction)):
        total_layout.append(layout_plot(num, img, predictions))

    return total_layout

        #
def parse_contents(contents, filename):

    aligned_faces = get_face(contents)
    #encoded_imges = [ arrtobase64(img) for img in aligned_faces]
    faces_arr=np.array(aligned_faces).reshape(len(aligned_faces),-1)
    y_prediction=session.run(y_pred, feed_dict={x: faces_arr })
    y_prediction = np.round(y_prediction,2)*100

    #import pdb;pdb.set_trace()

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
    images.extend(side_by_side_plot(aligned_faces,y_prediction))
    #import pdb;pdb.set_trace()
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
