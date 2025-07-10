# Cyberbullying Detection on Social Media Using Machine Learning

## 📌 Project Overview

This project focuses on building a **machine learning model to detect cyberbullying on social media platforms** based on the content of posts or messages. The system classifies text into different categories such as:

- **Age-based Cyberbullying**
- **Gender-based Cyberbullying**
- **Ethnicity-based Cyberbullying**
- **Other Cyberbullying**
- **Not Cyberbullying**

The project includes **data preprocessing, model training, hyperparameter tuning, model evaluation, and deployment using Flask**.

---

## 🚀 Project Purpose

- To **automate the detection of harmful or offensive messages** on social media.
- To help create a **safer online environment** by identifying posts that could be considered cyberbullying.
- To demonstrate the application of **machine learning in real-world text classification problems**.

---

## 📂 Project Structure

Project/
├── App/
│ ├── app.py
│ ├── model/
│ │ ├── model.pkl
│ │ └── fine_tune.pkl
│ ├── templates/
│ │ └── index.html
│ ├── static/
│ │ ├── main_css.css
│ │ └── main.js
│ └── routes/
│ └── init.py
├── Data/
│ ├── cyberbullying_balanced.csv
│ └── preprocessed_data.csv
├── Training/
│ ├── training_notebook.ipynb
│ └── preprocess_data.ipynb
├── Evaluation/
│ ├── evaluation_and_tuning.ipynb
│ └── best_model_saving.ipynb
├── README.md
├── requirements.txt
└── python_version.txt

yaml
Copy
Edit

---

## 🛠 Technologies & Tools Used

- **Python 3.x**
- **Scikit-learn (Machine Learning)**
- **Pandas & Numpy (Data handling)**
- **Flask (Web Framework)**
- **HTML, CSS, JavaScript (Frontend)**
- **Pickle (Model saving)**
- **Jupyter Notebook**

---

## 🔄 Workflow Steps

### 1️⃣ Data Preprocessing
- Clean text: remove links, symbols, stopwords, etc.
- Handle missing values (drop rows where tweet text is missing).
- Vectorize text using **TF-IDF**.

### 2️⃣ Model Training
- Train multiple models:  
  - Logistic Regression  
  - Support Vector Machine (SVM)  
  - Random Forest  
  - AdaBoost  
  - Gradient Boosting
- Evaluate and compare model accuracy.

### 3️⃣ Best Model Selection
- **SVM** selected as the best model (highest accuracy).
- Save model and TF-IDF vectorizer using **pickle**.

### 4️⃣ Hyperparameter Tuning
- Use **GridSearchCV** to improve SVM performance.
- Save the fine-tuned model as **fine_tune.pkl**.

### 5️⃣ Deployment
- Develop a simple **Flask web app** for real-time prediction.
- User inputs a message → model predicts → result shown on web page.

---

## ✅ Why SVM was Selected

- **Best accuracy** (78.02%) on the validation set.
- Performs well on **high-dimensional text data**.
- Handles **small and noisy datasets** effectively.

---

## ⚙ Requirements

Install all required packages using:

```bash
pip install -r requirements.txt
Typical requirements:

Copy
Edit
scikit-learn
pandas
numpy
flask
💡 How to Run the App
Preprocess data and train the model (Training/ folder).

Save the best model and fine-tuned model (Evaluation/ folder).

Run the Flask app:

bash
Copy
Edit
cd App
python app.py
Open your browser and visit:
http://127.0.0.1:5000/

📈 Sample Results
Algorithm	Accuracy
Logistic Regression	77.75%
SVM	78.02%
Random Forest	76.65%
AdaBoost	56.32%
Gradient Boosting	76.10%