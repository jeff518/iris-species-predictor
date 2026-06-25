
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import json

st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #6a0dad;
        text-align: center;
        margin-bottom: 2rem;
    }

    .prediction-card {
        background-color: #f0f8ff;
        padding: 2rem;
        border-radius: 10px;
        border-left: 5px solid #6a0dad;
        margin: 1rem 0;
    }

    .confidence-bar {
        height: 20px;
        background-color: #e0e0e0;
        border-radius: 10px;
        margin: 0.5rem 0;
    }

    .confidence-fill {
        height: 100%;
        border-radius: 10px;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        text-align: center;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(format_type='joblib'):
    """Load the model from the specified format"""
    try:
        if format_type == 'joblib':
            model = joblib.load('models/iris_model.joblib')

        elif format_type == 'pickle':
            with open('models/iris_model.pickle', 'rb') as f:
                model = pickle.load(f)

        return model

    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


@st.cache_resource
def load_model_info():
    """Load model metadata"""
    try:
        with open('models/model_info.json', 'r') as f:
            return json.load(f)

    except Exception as e:
        st.error(f"Error loading model info: {e}")
        return None


@st.cache_resource
def load_feature_ranges():
    """Load feature ranges for sliders"""
    try:
        with open('models/feature_ranges.json', 'r') as f:
            return json.load(f)

    except:
        return {
            'sepal_length': {'min': 4.0, 'max': 8.0, 'default': 5.8},
            'sepal_width': {'min': 2.0, 'max': 4.5, 'default': 3.0},
            'petal_length': {'min': 1.0, 'max': 7.0, 'default': 4.0},
            'petal_width': {'min': 0.1, 'max': 2.5, 'default': 1.2}
        }


# Load resources
model_info = load_model_info()
feature_ranges = load_feature_ranges()
model = load_model('joblib')


# Sidebar
with st.sidebar:

    st.title("⚙️ Settings")

    model_format = st.radio(
        "Model Format",
        ["joblib", "pickle"],
        help="Choose which serialized model format to use"
    )

    if st.button("🔄 Reload Model"):
        model = load_model(model_format)

        if model:
            st.success(
                f"Model loaded from {model_format} format!"
            )

    st.divider()

    st.subheader("📊 Model Information")

    if model_info:
        st.write(
            f"**Type:** {model_info.get('model_type', 'RandomForest')}"
        )

        st.write(
            f"**Accuracy:** {model_info.get('accuracy', 0.96):.1%}"
        )

        st.write(
            f"**Features:** {len(model_info.get('feature_names', []))}"
        )

        st.write(
            f"**Classes:** {len(model_info.get('target_names', []))}"
        )

    st.divider()

    st.subheader("🚀 Quick Actions")

    if st.button("📊 Show Dataset Info"):
        st.session_state.show_dataset_info = True

    if st.button("🎯 Make Prediction"):
        st.session_state.make_prediction = True


# Main Header
st.markdown(
    '<h1 class="main-header">🌸 Iris Flower Classification</h1>',
    unsafe_allow_html=True
)

st.markdown("""
This app predicts the species of an Iris flower based on its measurements
using a machine learning model.

Adjust the sliders below and click **Predict Species**.
""")


col1, col2 = st.columns([2, 1])

with col1:

    st.header("📝 Input Features")

    sepal_length = st.slider(
        "Sepal Length (cm)",
        min_value=float(feature_ranges['sepal_length']['min']),
        max_value=float(feature_ranges['sepal_length']['max']),
       value=float(
    feature_ranges['sepal_length'].get(
        'default',
        (feature_ranges['sepal_length']['min'] +
         feature_ranges['sepal_length']['max']) / 2
    )
),
        step=0.1
    )

    sepal_width = st.slider(
        "Sepal Width (cm)",
        min_value=float(feature_ranges['sepal_width']['min']),
        max_value=float(feature_ranges['sepal_width']['max']),
        value=float(
    feature_ranges['sepal_width'].get(
        'default',
        (feature_ranges['sepal_width']['min'] +
         feature_ranges['sepal_width']['max']) / 2
    )
),
        step=0.1
    )

    petal_length = st.slider(
        "Petal Length (cm)",
        min_value=float(feature_ranges['petal_length']['min']),
        max_value=float(feature_ranges['petal_length']['max']),
        value=float(
    feature_ranges['petal_length'].get(
        'default',
        (feature_ranges['petal_length']['min'] +
         feature_ranges['petal_length']['max']) / 2
    )
),
        step=0.1
    )

    petal_width = st.slider(
        "Petal Width (cm)",
        min_value=float(feature_ranges['petal_width']['min']),
        max_value=float(feature_ranges['petal_width']['max']),
       value=float(
    feature_ranges['petal_width'].get(
        'default',
        (feature_ranges['petal_width']['min'] +
         feature_ranges['petal_width']['max']) / 2
    )
),
        step=0.1
    )

with col2:

    st.header("📊 Current Values")

    features_df = pd.DataFrame({
        'Feature': [
            'Sepal Length',
            'Sepal Width',
            'Petal Length',
            'Petal Width'
        ],
        'Value (cm)': [
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]
    })

    st.dataframe(
        features_df,
        hide_index=True,
        use_container_width=True
    )


input_features = np.array([
    [sepal_length, sepal_width, petal_length, petal_width]
])


if st.button(
    "🎯 Predict Species",
    type="primary",
    use_container_width=True
):

    if model is not None and model_info is not None:

        try:

            prediction = model.predict(input_features)

            prediction_proba = model.predict_proba(
                input_features
            )[0]

            predicted_class = model_info[
                'target_names'
            ][prediction[0]]

            st.markdown(
                '<div class="prediction-card">',
                unsafe_allow_html=True
            )

            st.markdown(
                "### 🎯 Prediction Result"
            )

            st.markdown(
                f"**Predicted Species:** "
                f"**{predicted_class}**"
            )

            st.markdown(
                "### 📈 Confidence Scores"
            )

            for i, prob in enumerate(prediction_proba):

                species = model_info[
                    'target_names'
                ][i]

                percentage = prob * 100

                st.write(
                    f"**{species}: {percentage:.2f}%**"
                )

                st.progress(
                    float(prob)
                )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )

        except Exception as e:

            st.error(
                f"❌ Error making prediction: {e}"
            )

    else:

        st.error(
            "❌ Model could not be loaded."
        )


with st.expander("📚 About the Iris Dataset"):

    st.markdown("""
### Dataset Characteristics

- 150 samples
- 50 samples per species
- 4 numerical features
- 3 target classes

### Species

- Iris Setosa
- Iris Versicolor
- Iris Virginica

### Features

1. Sepal Length
2. Sepal Width
3. Petal Length
4. Petal Width

This application uses a Random Forest Classifier
trained on the Iris dataset.
""")


st.markdown("---")

st.markdown(
    """
    <div style='text-align:center'>
        <p>Built with Streamlit and Scikit-Learn by Jeffin John Rozario</p>
    </div>
    """,
    unsafe_allow_html=True
)

