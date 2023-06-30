import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

number_to_class = ['cardboard',
                   'glass',
                   'metal',
                   'paper',
                   'plastic',
                   'trash', ]

app = Flask(__name__)


def load_image(img_path_):
    img_ = image.load_img(img_path_, target_size=(200, 200))
    img_tensor = image.img_to_array(img_)  # (height, width, channels)
    # img_tensor = img_tensor.reshape(1, 50, 50, 3)
    img_tensor = np.expand_dims(img_tensor,
                                axis=0)  # (1, height, width, channels), add a dimension because the model expects
    # this shape: (batch_size, height, width, channels)
    img_tensor /= 255.  # imshow expects values in the range [0, 1]

    return img_tensor


def prediction(img_path_):
    new_image = load_image(img_path_)

    # Load the model
    model = load_model('model_200.h5')
    print("Model is loaded")

    pred = model.predict(new_image)
    pred_probability = np.max(pred[0], axis=-1)
    predicted_class = number_to_class[np.argmax(pred[0], axis=-1)]

    if pred_probability > 0.75:
        return_str = "The given image is classified as " + predicted_class + " with the probability of: " + \
                     str(pred_probability)
        return return_str, predicted_class
    else:
        return "The given image cannot be classified! Please try with different angle or size."


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("home.html")


@app.route("/about")
def about_page():
    return "This is the Waste classification Web Application!"


@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    global img_path, p, p_class
    if request.method == 'POST':
        img = request.files['my_image']

        img_path = "static/" + img.filename
        img.save(img_path)

        p, p_class = prediction(img_path)

    return render_template("home.html", prediction=p, pred_class=p_class, img_path=img_path)


if __name__ == "__main__":
    # app.debug = True
    app.run(debug=True)
