from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load Fine-Tuned Model
with open('model/fine_tune.pkl', 'rb') as f:
    tfidf, model = pickle.load(f)

# Define the label mapping (adjust according to your dataset labels)
label_mapping = {
    'not_cyberbullying': '✅ No Cyberbullying Detected',
    'age': '🚨 Cyberbullying Detected: Age-based',
    'gender': '🚨 Cyberbullying Detected: Gender-based',
    'ethnicity': '🚨 Cyberbullying Detected: Ethnicity-based',
    'religion': '🚨 Cyberbullying Detected: Religion-based',
    'other_cyberbullying': '🚨 Cyberbullying Detected: Other Type'
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        if message.strip() == '':
            return render_template('index.html', prediction="❗ Please enter a message.")
        
        vect_message = tfidf.transform([message])
        prediction = model.predict(vect_message)[0]
        
        result = label_mapping.get(prediction, "❗ Unknown Prediction")
        
        return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)

