import json
from flask import Flask, render_template, Response, request
import requests
from bs4 import BeautifulSoup
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image

model = model_from_json(open("fer.json", "r").read())
model.load_weights('fer.h5')

face_haar_cascade = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

app = Flask(__name__)

camera = cv2.VideoCapture(0)


def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces_detected = face_haar_cascade.detectMultiScale(
                gray_img, 1.32, 5)

            for (x, y, w, h) in faces_detected:
                print('WORKING')
                cv2.rectangle(frame, (x, y), (x+w, y+h),
                              (255, 0, 0), thickness=7)
                roi_gray = gray_img[y:y+w, x:x+h]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                img_pixels = image.img_to_array(roi_gray)
                img_pixels = np.expand_dims(img_pixels, axis=0)
                img_pixels /= 255

                predictions = model.predict(img_pixels)
                max_index = np.argmax(predictions[0])

                emotions = ['angry', 'disgust', 'fear',
                            'happy', 'sad', 'surprise', 'neutral']
                predicted_emotion = emotions[max_index]
                print(predicted_emotion)
                print(str(predictions[0]).replace(' ', ','))
                cv2.putText(frame, predicted_emotion, (int(x), int(y)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            resized_img = cv2.resize(frame, (1000, 700))

            ret, buffer = cv2.imencode('.jpg', frame)

            frame = buffer.tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def getEmotion():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces_detected = face_haar_cascade.detectMultiScale(
                gray_img, 1.32, 5)

            for (x, y, w, h) in faces_detected:
                print('WORKING')
                cv2.rectangle(frame, (x, y), (x+w, y+h),
                              (255, 0, 0), thickness=7)
                roi_gray = gray_img[y:y+w, x:x+h]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                img_pixels = image.img_to_array(roi_gray)
                img_pixels = np.expand_dims(img_pixels, axis=0)
                img_pixels /= 255

                predictions = model.predict(img_pixels)
                max_index = np.argmax(predictions[0])

                emotions = ['angry', 'disgust', 'fear',
                            'happy', 'sad', 'surprise', 'neutral']
                predicted_emotion = emotions[max_index]
                print(predicted_emotion)
                predictions = str(predictions[0]).replace(' ', ',')
                obj = [predicted_emotion, predictions]

                return obj


@app.route('/get_emotion_data')
def datares():
    obj = getEmotion()
    return Response(obj[1])


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    obj = getEmotion()
    angry = obj[1].split(',')[0].replace('[', '').replace(']', '')
    disgust = obj[1].split(',')[1].replace('[', '').replace(']', '')
    fear = obj[1].split(',')[2].replace('[', '').replace(']', '')
    happy = obj[1].split(',')[3].replace('[', '').replace(']', '')
    sad = obj[1].split(',')[4].replace('[', '').replace(']', '')
    surprise = obj[1].split(',')[5].replace('[', '').replace(']', '')
    neutral = obj[1].split(',')[6].replace('[', '').replace(']', '')

    return render_template('index.html', predicted_emotion=obj[0], angry=angry, disgust=disgust, fear=fear, happy=happy, sad=sad, surprise=surprise, neutral=neutral, pred=obj[1])


if __name__ == '__main__':
    app.run(debug=True)
