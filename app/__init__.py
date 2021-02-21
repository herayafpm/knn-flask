from flask import Flask, request, abort, jsonify
import numpy as np
import glob
from PIL import Image
import pickle
from os.path import isfile
from io import BytesIO
import base64


def test(datatest, clf):
    feature_datatest = []
    y = []
    for data in datatest:
        if 'kambing' in data:
            y.append(0)
        else:
            y.append(1)
        feature_datatest.append(extract_feature(data))
    X = np.vstack(feature_datatest)
    return clf.score(X, y)


def extract_feature(img):
    img = Image.open(img).convert('L')
    gl_0 = glcm(img, 0)
    gl_45 = glcm(img, 45)
    gl_90 = glcm(img, 90)
    gl_135 = glcm(img, 135)
    feature = np.array([
        np.average([contrast(gl_0), energy(gl_0),
                    homogenity(gl_0), entrophy(gl_0)]),
        np.average([contrast(gl_45), energy(gl_45),
                    homogenity(gl_45), entrophy(gl_45)]),
        np.average([contrast(gl_90), energy(gl_90),
                    homogenity(gl_90), entrophy(gl_90)]),
        np.average([contrast(gl_135), energy(gl_135),
                    homogenity(gl_135), entrophy(gl_135)])
    ])
    return feature


def extract_feature(img):
    img = Image.open(BytesIO(base64.b64decode(img))).convert('L')
    gl_0 = glcm(img, 0)
    gl_45 = glcm(img, 45)
    gl_90 = glcm(img, 90)
    gl_135 = glcm(img, 135)
    feature = np.array([
        np.average([contrast(gl_0), energy(gl_0),
                    homogenity(gl_0), entrophy(gl_0)]),
        np.average([contrast(gl_45), energy(gl_45),
                    homogenity(gl_45), entrophy(gl_45)]),
        np.average([contrast(gl_90), energy(gl_90),
                    homogenity(gl_90), entrophy(gl_90)]),
        np.average([contrast(gl_135), energy(gl_135),
                    homogenity(gl_135), entrophy(gl_135)])
    ])
    return feature


def contrast(matrix):
    width, height = matrix.shape
    res = 0
    for i in range(width):
        for j in range(height):
            res += matrix[i][j] * np.power(i-j, 2)
    return res


def energy(matrix):
    width, height = matrix.shape
    res = 0
    for i in range(width):
        for j in range(height):
            res += np.power(matrix[i][j], 2)
    return res


def homogenity(matrix):
    width, height = matrix.shape
    res = 0
    for i in range(width):
        for j in range(height):
            res += matrix[i][j] / (1+np.power(i-j, 2))
    return res


def entrophy(matrix):
    width, height = matrix.shape
    res = 0
    for i in range(width):
        for j in range(height):
            if matrix[i][j] > 0:
                res += matrix[i][j] * np.log2(matrix[i][j])
    return res


def glcm(img, degree):
    img = img.resize([128, 128], Image.NEAREST)
    arr = np.array(img)
    res = np.zeros((arr.max() + 1, arr.max() + 1), dtype=int)
    width, height = arr.shape
    if degree == 0:
        for i in range(width - 1):
            for j in range(height):
                res[arr[j, i+1], arr[j, i]] += 1
    elif degree == 45:
        for i in range(width - 1):
            for j in range(1, height):
                res[arr[j-1, i+1], arr[j, i]] += 1
    elif degree == 90:
        for i in range(width):
            for j in range(1, height):
                res[arr[j-1, i], arr[j, i]] += 1
    elif degree == 135:
        for i in range(1, width):
            for j in range(1, height):
                res[arr[j-1, i-1], arr[j, i]] += 1
    else:
        print("sudut tidak valid")
    return res


app = Flask(__name__)


@app.route('/')
@app.route('/index')
def index():
    return "Hello World"


@app.route('/proses', methods=['GET', 'POST'])
def proses():
    if(request.method == "POST"):
        data = request.json
        # datasets = []
        # datatests = []
        # for tipe in ['kambing', 'oplosan']:
        #     for i in range(0, 33):
        #         datasets.append("./data/{0}.{1}.jpg".format(tipe, i))
        #     for i in range(33, 50):
        #         datatests.append("./data/{0}.{1}.jpg".format(tipe, i))
        # datasets = [f for f in datasets if isfile(f)]
        # datatests = [f for f in datatests if isfile(f)]
        loaded_model = pickle.load(open('./finalized_model.sav', 'rb'))
        # result = test(datatests, loaded_model)
        predict = loaded_model.predict([extract_feature(data['img'])])
        if predict[0] == 0:
            result = "Kambing"
        else:
            result = "Oplosan"

        return jsonify({"result": result}), 200
    else:
        return "Hello World"


if __name__ == '__main__':
    app.run(debug=True)
