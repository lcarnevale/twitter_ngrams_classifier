# thierd parties libraries
from flask import Flask
from flask_restful import Api
from resources.predict import Predict

app = Flask(__name__)
api = Api(app)

api.add_resource(Predict, '/')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
