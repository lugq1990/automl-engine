"""Main entry for this project restful API."""

import numpy as np
import json
from flask import request, Flask, jsonify
from flask_restful import Resource, Api

from automl.estimator import ClassificationAutoML


app = Flask(__name__)
api = Api(app)


@app.before_first_request
def load_models():
    auto_classification = ClassificationAutoML.reconstruct()

    app.predictor = auto_classification


def get_request_data(request):
    """To get used data from request."""
    json_data = request.get_json(force=True)
        
    try:
        data = json.loads(json_data)['data']
    except Exception as e:
        return "Not get data correctly!", 403

    if not isinstance(data, list):
        return "Prediction data should a type of list", 403
    
    # convert data into array type
    try:
        data = np.array(data)
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
    except Exception as e:
        raise ValueError("When try to convert data into array type, we get error: {}".format(e))

    return data


class Predict(Resource):
    """Main logic for real prediction logic that we need to use for restful API call.
    By default, just provide with request with json string: json=`{"data":[[data], [data]]}`
    """
    def post(self):
        data = get_request_data(request)
        # just use the re-construct object to get prediction based on best trained models!
        pred = app.predictor.predict(x=data)
        
        if len(pred) == 0 or pred is None:
            return jsonify({"result": "We couldn't get prediction for your data!"})

        result = {"prediction": pred.tolist()}
        return jsonify(result)


class PredictProb(Resource):
    def post(self):
        data = get_request_data(request)

        prob = app.predictor.predict_proba(x=data)

        if len(prob) == 0 or prob is None:
            return jsonify({"result": "We couldn't get probability for your data!"})
        
        result = {"probability": prob.tolist()}

        return jsonify(result)


api.add_resource(Predict, '/predict')
api.add_resource(PredictProb, '/predict_proba')


if __name__ == '__main__':
    app.run(debug=True)
