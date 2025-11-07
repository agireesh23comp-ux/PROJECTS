import cv2
import sys
import numpy as np
from tensorflow import keras

MODEL_PATH = "tf-cnn-model-new.h5"

def predict_digit(image_path):
    
    # load model
    model = keras.models.load_model(MODEL_PATH)
    print("[INFO] Loaded model from disk.")

    image = cv2.imread(image_path, 0)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    image1 = cv2.resize(image, (28, 28))    # Resize to 28x28
    image2 = image1.reshape(1, 28, 28, 1)
    
    # Normalize pixel values to 0-1 range if needed
    image2 = image2.astype('float32') / 255.0

    cv2.imshow('digit', image1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    pred = np.argmax(model.predict(image2, verbose=0), axis=-1)
    return pred[0]    

def main(image_path):
    predicted_digit = predict_digit(image_path)
    print('Predicted Digit: {}'.format(predicted_digit))
 
if __name__ == "__main__":
    try:
        main(image_path=sys.argv[1])
    except IndexError:
        print('[ERROR]: Please provide image path as argument')
        print('Usage: python script.py <image_path>')
    except ValueError as e:
        print(f'[ERROR]: {e}')
    except Exception as e:
        print(f'[ERROR]: {e}')