# 🌸 Iris Flower Classification

An interactive Machine Learning web application built with Streamlit that predicts the species of an Iris flower based on its physical measurements.

## 📌 Overview

This project uses a trained Machine Learning model to classify Iris flowers into one of three species:

* Iris Setosa
* Iris Versicolor
* Iris Virginica

Users can adjust flower measurements using sliders and instantly receive a prediction from the model.

## 🚀 Features

* Interactive Streamlit interface
* Real-time prediction
* Multiple model format support (Joblib and Pickle)
* Dataset information display
* Model statistics and accuracy visualization
* Responsive and user-friendly design

## 🛠️ Technologies Used

* Python
* Streamlit
* Scikit-Learn
* Pandas
* NumPy
* Joblib
* Pickle

## 📊 Dataset

This project uses the famous Iris Dataset, which contains measurements of iris flowers from three different species.

Features used:

1. Sepal Length (cm)
2. Sepal Width (cm)
3. Petal Length (cm)
4. Petal Width (cm)

Target Classes:

* Setosa
* Versicolor
* Virginica

## 🤖 Machine Learning Model

Model Type: Random Forest Classifier

The model was trained on the Iris dataset and achieves approximately 90% accuracy on the test set.

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/iris-flower-classification.git
cd iris-flower-classification
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## 🎯 Usage

1. Open the application in your browser.
2. Adjust the flower measurements using the sliders.
3. Click **Predict Species**.
4. View the predicted Iris species and model output.


## 🔮 Future Improvements

* Probability visualization
* Species image display
* Model comparison dashboard
* Dark mode support
* Explainable AI (SHAP/LIME) integration

## 👨‍💻 Author

Developed as a Machine Learning and Streamlit project for learning classification models and interactive AI application development.


