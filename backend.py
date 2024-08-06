
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

with open('logistic_regression_model.pkl', 'rb') as f:
    logistic_regression_model = pickle.load(f)

with open('naive_bayes_model.pkl', 'rb') as f:
    naive_bayes_model = pickle.load(f)

from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

logistic_regression_model_path = r"C:/Users/Dell/Downloads/logistic_regression_model.pkl"
naive_bayes_model_path = r"C:/Users/Dell/Downloads/naive_bayes_model.pkl"
decision_tree_model_path = r"C:/Users/Dell/Downloads/decision_tree_model.pkl"
vectorizer_path = r"C:/Users/Dell/Downloads/tfidf_vectorizer.pkl"


with open(logistic_regression_model_path, 'rb') as file:
    loaded_log_reg = pickle.load(file)

with open(naive_bayes_model_path, 'rb') as file:
    loaded_naive_bayes = pickle.load(file)

with open(decision_tree_model_path, 'rb') as file:
    loaded_decision_tree = pickle.load(file)

with open(vectorizer_path, 'rb') as file:
    loaded_vectorizer = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']
    model_name = data['model']

    if not text:
        return jsonify({'error': 'No input text provided'}), 400

    text_vectorized = loaded_vectorizer.transform([text])

    prediction = None
    if model_name == 'logistic_regression':
        prediction = loaded_log_reg.predict(text_vectorized)
    elif model_name == 'naive_bayes':
        prediction = loaded_naive_bayes.predict(text_vectorized)
    elif model_name == 'decision_tree':
        prediction = loaded_decision_tree.predict(text_vectorized)
    else:
        return jsonify({'error': 'Invalid model selected'}), 400

    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
