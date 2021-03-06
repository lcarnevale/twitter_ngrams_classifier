# standard libraries
import pickle
# local libraries
from common.posts_reader import postprocess
import common.classifier as clf
# third parties libraries
import pandas as pd
from nltk import FreqDist
from flask import request
from flask_restful import Resource

class Classify(Resource):
    """
    """
    def post(self):
        """
        """
        json_data = request.get_json(force=True)

        try:
            utterances = json_data['utterances']
            utterances = list(set(utterances))
        except KeyError as e:
            data['error'] = 'Malformed JSON format: %s' % (e)
            status_code = 400
            return

        samples = utterances
        df = pd.Series(samples)

        labels = clf.predict(df, 'models/MultinomialNB.pickle')
        return postprocess(samples, labels)
