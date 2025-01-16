from flask import Flask, render_template, request
import os


from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
import cv2
import mediapipe as mp
import keras
from keras.layers import Dense
import numpy as np
import base64

counter = 0
alpha = 'A'

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# define the flask app
app = Flask(__name__)

# Load and preprocess the dataset
dataset = np.load('X.npy')  # Adjust path as needed
X = dataset  # Features (64x64 images)

# Create placeholder labels (for demonstration purposes)
if X.shape[0] > 0:
    y = np.random.randint(0, 25, X.shape[0])  # Generate random labels for 25 classes
else:
    raise ValueError("Dataset is empty")

# Normalize and reshape the dataset
X = X.reshape(-1, 64, 64, 1).astype('float32') / 255  # Normalize pixel values

# Define the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(64, 64, 1)),
    keras.layers.Dense(380, activation='relu'),
    keras.layers.Dense(200, activation='relu'),
    keras.layers.Dense(25, activation='softmax')
])

model.compile(optimizer='adam', loss=keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
model.fit(X, y, epochs=3, batch_size=32)
print("Model training completed")

# Utility functions
def applyHandPointsVideo(img_path):
    IMAGE_FILES = [img_path]
    global alpha, counter
    try:
        with mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=2,
                min_detection_confidence=0.5) as hands:
            for idx, file in enumerate(IMAGE_FILES):
                # Read an image, flip it around y-axis for correct handedness output (see
                # above).
                image = cv2.imread(file)
                # Convert the BGR image to RGB before processing.
                results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                # Print handedness and draw hand landmarks on the image.
                print('Handedness:', results.multi_handedness)
                if not results.multi_hand_landmarks:
                    continue
                image_height, image_width, _ = image.shape
                img = np.zeros((image_height, image_width, 3), dtype=np.uint8)
                annotated_image = image.copy()
                for hand_landmarks in results.multi_hand_landmarks:
                    x = [landmark.x for landmark in hand_landmarks.landmark]
                    y = [landmark.y for landmark in hand_landmarks.landmark]

                    center = np.array([np.mean(x) * img.shape[1], np.mean(y) * img.shape[0]]).astype('int32')
                    start_point = (center[0] - 100, center[1] - 100)
                    end_point = (center[0] + 100, center[1] + 100)

                    mp_drawing.draw_landmarks(
                        annotated_image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                    mp_drawing.draw_landmarks(
                        img,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=2))

                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                print(img.shape)
                print(start_point, end_point)
                if (img.shape[0] > 200 and img.shape[1] > 200):
                    if (start_point[0] < 0):
                        start_point = (0, start_point[1])
                    if (start_point[1] < 0):
                        start_point = (start_point[0], 0)
                    if (end_point[0] < 0):
                        end_point = (0, end_point[1])
                    if (end_point[1] < 0):
                        end_point = (end_point[0], 0)
                    img = img[start_point[1]:end_point[1], start_point[0]:end_point[0]]
                print(img.shape)
                print(counter)
                imgPath = 'images/marked/Y/img' + str(counter) + '.png'
                skeletonPath = 'images/skeleton/Y/img' + str(counter) + '.png'
                cv2.imwrite(imgPath, annotated_image)
                cv2.imwrite(skeletonPath, img)
                counter += 1
                return skeletonPath
    except:
        return "Hand not detected"

def applyHandPointsUpload(img_path):
    IMAGE_FILES = [img_path]
    global alpha, counter
    try:
        with mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=2,
                min_detection_confidence=0.5) as hands:
            for idx, file in enumerate(IMAGE_FILES):
                # Read an image, flip it around y-axis for correct handedness output (see
                # above).
                image = cv2.imread(file)
                # Convert the BGR image to RGB before processing.
                results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                # Print handedness and draw hand landmarks on the image.
                print('Handedness:', results.multi_handedness)
                if not results.multi_hand_landmarks:
                    continue
                image_height, image_width, _ = image.shape
                img = np.zeros((image_height, image_width, 3), dtype=np.uint8)
                annotated_image = image.copy()
                for hand_landmarks in results.multi_hand_landmarks:
                    x = [landmark.x for landmark in hand_landmarks.landmark]
                    y = [landmark.y for landmark in hand_landmarks.landmark]

                    center = np.array([np.mean(x) * img.shape[1], np.mean(y) * img.shape[0]]).astype('int32')
                    start_point = (center[0] - 150, center[1] - 150)
                    end_point = (center[0] + 150, center[1] + 150)

                    mp_drawing.draw_landmarks(
                        annotated_image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                    mp_drawing.draw_landmarks(
                        img,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=4),
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4))

                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                print(img.shape)
                print(start_point, end_point)
                if (img.shape[0] > 300 and img.shape[1] > 300):
                    if (start_point[0] < 0):
                        start_point = (0, start_point[1])
                    if (start_point[1] < 0):
                        start_point = (start_point[0], 0)
                    if (end_point[0] < 0):
                        end_point = (0, end_point[1])
                    if (end_point[1] < 0):
                        end_point = (end_point[0], 0)
                    img = img[start_point[1]:end_point[1], start_point[0]:end_point[0]]
                print(img.shape)
                print(counter)
                imgPath = 'images/marked/test/img' + str(counter) + '.png'
                skeletonPath = 'images/skeleton/test/img' + str(counter) + '.png'
                cv2.imwrite(imgPath, annotated_image)
                cv2.imwrite(skeletonPath, img)
                counter += 1
                return skeletonPath
    except:
        return "Hand not detected"

ImagesCaptured = np.empty(shape=(1,300,300,1), dtype=np.uint8)
# Labels mapping to indices
labels = ['1','2','3','4','5','6','7','8','9','0']

ImagesCaptured = np.empty(shape=(1, 64, 64, 1), dtype=np.uint8)

# Labels mapping to indices
labels = [chr(i) for i in range(65, 91)]  # A-Z

def model_predict(img_path, function):
    global ImagesCaptured, labels
    path = function(img_path)
    print("Path : ", path)
    if path == "Hand not detected" or path is None:
        return "Unable to detect hand"

    # Load and preprocess the image
    test_image = image.load_img(path, target_size=(64, 64), color_mode='grayscale')
    test_image = image.img_to_array(test_image).astype('float32') / 255
    test_image = test_image.reshape(1, 64, 64, 1)

    # Predict the label
    out = model.predict(test_image)
    pred = np.argmax(out, axis=1)
    pred = labels[pred[0]]
    return pred

# Flask routes
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from the POST request
        f = request.files['file']
        # Save the file to the uploads folder
        basepath = os.path.dirname(os.path.realpath('__file__'))
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        result = model_predict(file_path, applyHandPointsUpload)
        return 'Predicted alphabet : ' + result
    return None

@app.route('/predict-img', methods=['GET', 'POST'])
def predictImg():
    if request.method == 'POST':
        basepath = os.path.dirname(os.path.realpath('__file__'))
        file_path = os.path.join(basepath, 'uploads', secure_filename('videoImg.png'))
        with open(file_path, "wb") as fh:
            fh.write(base64.decodebytes(request.data))

        # Make prediction
        result = model_predict(file_path, applyHandPointsVideo)
        return 'Predicted alphabet : ' + result
    return None



if __name__ == '__main__':
    app.run(debug=True, port=5926)
