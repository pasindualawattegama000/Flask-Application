from flask import Flask, render_template, request

# from keras.preprocessing.image import load_img
# from keras.preprocessing.image import img_to_array

import tensorflow as tf
import cv2

model = tf.keras.models.load_model(
    'C:/Users/pasin/Desktop/ML website/Model/my_model.keras')

# Show the model architecture
# model.summary()

app = Flask(__name__)


@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict():
    # name of the image input in the html
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    randomImg = cv2.imread(image_path)
    randomImgR = cv2.resize(randomImg, (100, 100))
    randomImgR = randomImgR.reshape(1, 100, 100, 3)

    pred = model.predict(randomImgR)
    # print(pred)
    # print("Probability Cat: ", pred[0, 0])
    # print("Probability Dod: ", pred[0, 1])
    if pred[0, 0] > pred[0, 1]:
        result = 'Cat'
    else:
        result = 'Dog'

    # print("The model says its a : ", result)

    return render_template('index.html', prediction=result)


if __name__ == '__main__':
    app.run(port=3000, debug=True)
