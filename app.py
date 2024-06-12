import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

model = joblib.load('models/itis_grant_model.pkl')


model_grant = joblib.load('models/itis_grant_model.pkl')
model_student_performance = joblib.load('models/student_performance_model.pkl')



@app.route('/grant_predict', methods=['POST'])
def grant_predict():
    data = request.json
    average_score = data['average_score']
    grant_student_count = data['grant_student_count']
    grant_student_applied = data['grant_student_applied']

    input_data = [[average_score, grant_student_count, grant_student_applied]]

    prediction = model_grant.predict(input_data)

    response = {'prediction': prediction[0]}
    return jsonify(response), 200


@app.route('/predict_student_score', methods=['POST'])
def predict_student_score():
    data = request.json
    m_edu = data['mEdu']
    f_edu = data['fEdu']
    study_time = data['studyTime']
    failures = data['failures']
    support = data['support']
    higher = data['higher']
    absences = data['absences']

    input_data = [[m_edu, f_edu, study_time, failures, support, higher, absences]]
    prediction = model_student_performance.predict(input_data)

    response = {'score': prediction[0]}
    return jsonify(response), 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
