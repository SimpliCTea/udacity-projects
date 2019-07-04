import os
from flask import render_template, jsonify, request, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np
from keras.preprocessing import image

from app import app

from app.breed_classifier import predict

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

RESPONSE_MSG_DICT = {
    'dog': ('<p class="result pb-2 bb">Yep, I can see a dog. I\'d say it\'s a :</p>'
        '<p class="result-strong"><span>{}</span></p>'
        '<p class="result pt-2 bt"> Was I right? :P</p>'),
    'human': ('<p class="result pb-2 bb">Well, this is not a dog. I\'d say the person in this '
        'image most closely resembles a:</p>'
         '<p class="result-strong"><span>{}</span></p>'
         '<p class="result pt-2 bt">Can you see it too?</p>'),
    'undefined': ('<p class="result">I\'m sorry, this may be me, but I don\'t see a dog '
        'or human in this image. Please make sure to upload an image '
        'of a dog or a human with clearly visible face. That makes it '
        'easier for me.</p>')
}

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template("index.html")


@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/references')
def references():
    return render_template("references.html")

@app.route('/classify', methods=['GET','POST'])
def classify():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'img' not in request.files:
            print('No file part')
            return redirect(request.url)
        img = request.files['img']
        # if user does not select file, browser also
        # submit a empty part without filename
        if img.filename == '':
            print('No selected file')
            return redirect(request.url)
        if img and allowed_file(img.filename):
            # get image and store it temporarily
            filename = secure_filename(img.filename)
            print('[VIEWS]: Received image: {}'.format(filename))
            filepath = os.path.join(app.instance_path, filename)
            img.save(os.path.join(app.instance_path, filename))
            print('[VIEWS]: Saved image under: {}'.format(filepath))
            # set up response
            resp = {
                'success': False,
                'message': '',
                'data': {}
            }
            # try to get the prediction on the image
            try:
                resp['data'] = predict(filepath)
                resp['success'] = True
                resp['message'] = RESPONSE_MSG_DICT[resp['data']['race']].format(resp['data']['breed'])
            # catch AssertionError as this means the algorithm couldn't detect a dog or human face in the image
            except AssertionError:
                resp['message'] = RESPONSE_MSG_DICT['undefined']
            # remove image again, we don't want to save all user images after all
            os.remove(filepath)
            print('[VIEWS]: Deleted image: {}'.format(filename))
            # return the prediction and the result string as json
            return jsonify(resp)
        