import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import cv2
import numpy as np
from tensorflow import keras
import base64
from io import BytesIO
from PIL import Image

MODEL_PATH = "tf-cnn-model.h5"

# Load model once at startup
model = keras.models.load_model(MODEL_PATH, compile=False)
print("[INFO] Model loaded successfully!")

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                html.H1("✍️ Handwritten Digit Recognition", 
                       className="text-center my-5 fw-bold text-primary"),
                width=12
            )
        ),
        
        dbc.Row(
            dbc.Col(
                dcc.Upload(
                    id='upload-image',
                    children=html.Div([
                        html.I(className="fas fa-cloud-upload-alt", 
                              style={'fontSize': '3rem', 'color': '#0d6efd', 'marginBottom': '1rem'}),
                        html.Br(),
                        html.P("Drag and drop or click to upload an image", 
                              style={'fontSize': '1.1rem', 'color': '#666'}),
                        html.P("(PNG, JPG, JPEG)", 
                              style={'fontSize': '0.9rem', 'color': '#999'})
                    ]),
                    style={
                        'width': '100%',
                        'height': '300px',
                        'lineHeight': '300px',
                        'borderWidth': '3px',
                        'borderStyle': 'dashed',
                        'borderRadius': '10px',
                        'textAlign': 'center',
                        'margin': 'auto',
                        'padding': '20px',
                        'boxSizing': 'border-box',
                        'backgroundColor': '#f8f9fa',
                        'cursor': 'pointer',
                        'transition': 'all 0.3s ease'
                    },
                    multiple=False
                ),
                width=10, lg=8, md=10,
                className="mx-auto"
            )
        ),
        html.Br(),
        
        dbc.Row(
            dbc.Col(
                html.Div(id='output-div'),
                width=10, lg=8, md=10,
                className="mx-auto"
            )
        ),
    ],
    fluid=True,
    style={'backgroundColor': '#ffffff', 'minHeight': '100vh', 'paddingTop': '20px'}
)

def predict_digit(image_array):
    """Predict digit from image array"""
    # Ensure it's grayscale
    if len(image_array.shape) == 3:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    
    # Resize to 28x28
    image_resized = cv2.resize(image_array, (28, 28))
    
    # Reshape for model
    image_input = image_resized.reshape(1, 28, 28, 1)
    
    # Normalize
    image_input = image_input.astype('float32') / 255.0
    
    # Predict
    pred = np.argmax(model.predict(image_input, verbose=0), axis=-1)
    return pred[0]

@app.callback(
    Output('output-div', 'children'),
    Input('upload-image', 'contents'),
    State('upload-image', 'filename'),
    prevent_initial_call=True
)
def update_output(contents, filename):
    if contents is None:
        return html.Div()
    
    try:
        # Decode the image
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        image = Image.open(BytesIO(decoded))
        image_array = np.array(image)
        
        # Predict digit
        predicted_digit = predict_digit(image_array)
        
        # Convert image to base64 for display
        img_pil = Image.fromarray(image_array)
        buffer = BytesIO()
        img_pil.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    dbc.Col(
                        html.Div([
                            html.P("Uploaded Image:", className="fw-bold text-secondary mb-3"),
                            html.Img(
                                src=f'data:image/png;base64,{img_base64}',
                                style={'maxWidth': '100%', 'height': 'auto', 'borderRadius': '8px'}
                            )
                        ]),
                        width=12, md=6,
                        className="text-center mb-4 mb-md-0"
                    ),
                    dbc.Col(
                        html.Div([
                            html.P("Identifying...", 
                                  id='identifying-text',
                                  className="text-info fw-bold mb-4",
                                  style={'fontSize': '1.2rem', 'animation': 'pulse 1.5s infinite'}),
                            html.Div([
                                html.P("Predicted Digit:", className="text-secondary fw-bold mb-2"),
                                html.Div(
                                    str(predicted_digit),
                                    style={
                                        'fontSize': '4rem',
                                        'fontWeight': 'bold',
                                        'color': '#0d6efd',
                                        'textAlign': 'center',
                                        'padding': '20px',
                                        'backgroundColor': '#e7f1ff',
                                        'borderRadius': '10px',
                                        'border': '3px solid #0d6efd'
                                    }
                                )
                            ])
                        ]),
                        width=12, md=6,
                        className="d-flex flex-column justify-content-center"
                    )
                ])
            ])
        ], className="shadow-sm border-0")
        
    except Exception as e:
        return dbc.Alert(
            f"Error processing image: {str(e)}", 
            color="danger",
            className="mt-3"
        )

if __name__ == '__main__':
    app.run(debug=True)