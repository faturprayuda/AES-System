from flask import Flask, send_file
from flask_restful import Api, Resource, reqparse
import werkzeug
import os.path
from os import path
from nn import *
from plag import *
from modelNN import *

app = Flask(__name__)
api = Api(app)
dir_path = os.path.dirname(os.path.realpath(__file__))
parse = reqparse.RequestParser()

def loadFile(codeFile):
    parse.add_argument(
        'file', type=werkzeug.datastructures.FileStorage, location='files')
    parse.add_argument(
        'user_id', type=int, location='form')
    args = parse.parse_args()
    data_file = args['file']
    user_id = str(args['user_id'])
    if codeFile == 1 :
        PathFile = user_id + "/file_kunci_jawaban.xlsx"
    else :
        PathFile = user_id + "/file_jawaban.xlsx"
    if path.exists(user_id):
        data_file.save(PathFile)
    else:
        pathDir = os.path.join(dir_path, user_id)
        os.mkdir(pathDir)
        data_file.save(PathFile)
    if codeFile == 1 :
        file = MakeModel(PathFile, user_id)
    else :
        file = getFile(PathFile, user_id)
    return {'status': 'success', 'data': file}


def plagiarism() :
    parse.add_argument(
        'file', type=werkzeug.datastructures.FileStorage, location='files')
    parse.add_argument(
        'user_id', type=int, location='form')
    args = parse.parse_args()
    data_file = args['file']
    user_id = str(args['user_id'])
    PathFile = user_id + "/file_kunci_jawaban.xlsx"
    if path.exists(user_id):
        data_file.save(PathFile)
    else:
        pathDir = os.path.join(dir_path, user_id)
        os.mkdir(pathDir)
        data_file.save(PathFile)
    file = checkPlag(PathFile)
    return {'status': 'success', 'data': file}


class UploadKeyFile(Resource):
    def post(self):
        return loadFile(1)

class UploadFile(Resource):
    def post(self):
        return loadFile(2)

class checkPlagiarism(Resource):
    def post(self):
        return plagiarism()

api.add_resource(UploadKeyFile, "/upload-kunci-jawaban")
api.add_resource(UploadFile, "/upload-jawaban")
api.add_resource(checkPlagiarism, "/check-plagiarism")


if __name__ == "__main__":
    app.run(debug=True)
