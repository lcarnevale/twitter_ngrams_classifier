# thierd parties libraries
from flask import Flask
from flask_restful import Api
from resources.classify import Classify


app = Flask(__name__)
api = Api(app)

api.add_resource(Classify, '/')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5002', debug=True)
