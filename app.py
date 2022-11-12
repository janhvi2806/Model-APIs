from flask import Flask, request, jsonify
from prediction import Prediction

app = Flask(__name__)

model_obj = Prediction()


@app.route('/predict', methods=["POST"])
def predict():
    # return "done"
    content = request.json
    user_data = content["user_data"]
    model_output = model_obj.predict(user_data)
    return jsonify({"output": f"{model_output}"})


if __name__ == "__main__":
    app.run(debug=True)
