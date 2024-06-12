import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

model = joblib.load('models/itis_grant_model.pkl')


@app.route('/grant_predict', methods=['POST'])
def grant_predict():
    data = request.json
    average_score = data['average_score']
    grant_student_count = data['grant_student_count']
    grant_student_applied = data['grant_student_applied']

    input_data = [[average_score, grant_student_count, grant_student_applied]]

    prediction = model.predict(input_data)

    response = {'prediction': prediction[0]}
    return jsonify(response), 200


if __name__ == '__main__':
    app.run(debug=True)
