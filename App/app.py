from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the fine-tuned model and TF-IDF vectorizer
with open('model/fine_tune.pkl', 'rb') as f:
    tfidf, model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']

        if not message.strip():
            return render_template('index.html', prediction="❗ Please enter a message.", message=message)

        # Vectorize and predict
        vect = tfidf.transform([message])
        prediction = model.predict(vect)[0]

        label_mapping = {
            'not_cyberbullying': '✅ No Cyberbullying Detected',
            'age': '🚨 Age-based Cyberbullying',
            'ethnicity': '🚨 Ethnicity-based Cyberbullying',
            'gender': '🚨 Gender-based Cyberbullying',
            'religion': '🚨 Religion-based Cyberbullying',
            'other_cyberbullying': '🚨 Other Types of Cyberbullying'
        }

        result = label_mapping.get(prediction, "⚠️ Unable to classify")

        return render_template('index.html', prediction=result, message=message)

if __name__ == '__main__':
    app.run(debug=True)
