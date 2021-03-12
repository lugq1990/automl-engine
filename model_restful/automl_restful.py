"""Main entry for this project restful API."""

import numpy as np
import json
from flask import request, Flask, jsonify
from flask_restful import Resource, Api

from auto_ml.automl import ClassificationAutoML


app = Flask(__name__)
api = Api(app)


@app.before_first_request
def load_models():
    auto_classification = ClassificationAutoML.reconstruct()

    app.predictor = auto_classification



class Predict(Resource):
    """Main logic for real prediction logic that we need to use for restful API call."""
    def post(self):
        json_data = request.get_json(force=True)
        
        try:
            data = json.loads(json_data)['data']
        except Exception as e:
            return "Not get data correctly!", 403

        if not isinstance(data, list):
            return "Prediction data should a type of list", 403
        
        # convert data into array type
        data = np.array(data)

        # just use the re-construct object to get prediction based on best trained models!
        pred = app.predictor.predict(x=data)

        result = {"prediction": pred.tolist()}
        return jsonify(result)


api.add_resource(Predict, '/predict')


if __name__ == '__main__':
    app.run(debug=False)
