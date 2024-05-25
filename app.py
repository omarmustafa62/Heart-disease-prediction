import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
app = Flask(__name__)
model = pickle.load(open('model2.pkl', 'rb'))
@app.route('/')
def home():
    return render_template('home.html')
@app.route("/pred.html")
def find():
		return render_template("pred.html")
@app.route('/predict' ,methods=['POST'])
def predict():
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model.predict(final_features)
    if prediction == 0:
        return render_template('result.html', prediction_text="The test results indicate no significant health issues")
    else:
        return render_template('result.html', prediction_text="Your tests have shown some abnormal results, and it is important to follow the suggested medical guidance for further follow-up") 
@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])
    output = prediction[0]
    return jsonify(output)
if __name__ == "__main__":
    app.run(debug=True)