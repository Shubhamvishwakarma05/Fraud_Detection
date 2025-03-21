import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration for a professional look
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide", page_icon="💳")

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {background-color: #f5f5f5;}
    .stButton>button {background-color: #4CAF50; color: white; border-radius: 10px;}
    .stTextInput>div>input {border-radius: 10px;}
    .sidebar .sidebar-content {background-color: #ffffff;}
    h1 {color: #2c3e50;}
    h3 {color: #34495e;}
    .contact-icon {vertical-align: middle; margin-right: 5px;}
    </style>
""", unsafe_allow_html=True)


# Load and preprocess data
@st.cache_data
def load_data():
    data = pd.read_csv("creditcard.csv")  # Replace with your dataset path
    return data


# Train the model
@st.cache_resource
def train_model(data):
    X = data.drop(columns=["Class"])
    y = data["Class"]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

    # Train XGBoost model
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train_balanced, y_train_balanced)

    # Predictions and evaluation
    y_pred = model.predict(X_test_scaled)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    return model, scaler, X_test_scaled, y_test, report, cm


# Main app
def main():
    # Load data
    data = load_data()

    # Sidebar for navigation and contact support
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Model Training", "Prediction", "Dataset Info"])

    # Contact Support in Sidebar
    st.sidebar.title("Contact Support")
    st.sidebar.markdown("""
        If you have any issues or questions, feel free to reach out!  

        <a href="https://www.linkedin.com/in/shubhamvishwakarma05/" target="_blank">
            <img src="https://img.icons8.com/ios-filled/20/000000/linkedin.png" class="contact-icon"/> LinkedIn
        </a>  
        <br>
        <a href="https://github.com/Shubhamvishwakarma05" target="_blank">
            <img src="https://img.icons8.com/ios-filled/20/000000/github.png" class="contact-icon"/> GitHub
        </a>
    """, unsafe_allow_html=True)

    # Home Page
    if page == "Home":
        st.title("💳 Credit Card Fraud Detection")
        st.markdown("""
            Welcome to the Credit Card Fraud Detection App!  
            This tool uses an XGBoost model to predict fraudulent transactions based on a dataset of credit card transactions.  
            Navigate using the sidebar to train the model, make predictions, or learn about the dataset.
        """)
        st.image("creditcardfrauddetection.jpg", use_container_width=True)

    # Model Training Page
    elif page == "Model Training":
        st.title("Model Training")
        st.write("Training the XGBoost model on the dataset...")

        # Train model
        model, scaler, X_test_scaled, y_test, report, cm = train_model(data)

        # Display evaluation metrics
        st.subheader("Model Performance")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Classification Report:")
            st.json(report)
        with col2:
            st.write("Confusion Matrix:")
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            st.pyplot(fig)

    # Prediction Page
    elif page == "Prediction":
        st.title("Fraud Prediction")
        st.write("Enter transaction details to predict if it's fraudulent.")

        # Input fields for features
        st.subheader("Transaction Details")
        cols = st.columns(4)
        inputs = []
        for i, col in enumerate(data.columns[:-1]):  # Exclude 'Class'
            with cols[i % 4]:
                inputs.append(st.number_input(col, value=0.0, step=0.01))

        # Predict button
        if st.button("Predict"):
            # Prepare input data
            input_data = np.array(inputs).reshape(1, -1)
            model, scaler, _, _, _, _ = train_model(data)
            input_scaled = scaler.transform(input_data)

            # Make prediction
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1]

            # Display result
            st.subheader("Prediction Result")
            if prediction == 1:
                st.error(f"⚠️ Fraudulent Transaction Detected! (Probability: {probability:.2%})")
            else:
                st.success(f"✅ Legitimate Transaction (Probability of Fraud: {probability:.2%})")

            # Visualization
            fig, ax = plt.subplots()
            ax.pie([probability, 1 - probability], labels=['Fraud', 'Legit'], autopct='%1.1f%%',
                   colors=['#ff9999', '#66b3ff'])
            st.pyplot(fig)

    # Dataset Info Page
    elif page == "Dataset Info":
        st.title("Dataset Information")
        st.markdown("""
            This app uses a credit card transaction dataset to predict fraudulent activities. The dataset contains transactions with anonymized features processed using PCA (Principal Component Analysis) for privacy, along with 'Time', 'Amount', and 'Class' columns. Below is an explanation of each feature and its role in fraud detection:
        """)

        st.subheader("Features Explanation")
        st.markdown("""
            - **Time**: The seconds elapsed between each transaction and the first transaction in the dataset.  
              *Role*: Helps identify patterns over time (e.g., fraud might occur in bursts or at specific intervals).
            - **V1 to V28**: Principal components obtained from PCA transformation of original features (e.g., cardholder details, merchant info). These are numerical and anonymized for security.  
              *Role*: Capture hidden patterns and relationships in the data that differentiate legitimate and fraudulent transactions. For example, unusual combinations of these values might indicate fraud.
            - **Amount**: The transaction amount in the local currency.  
              *Role*: Fraudulent transactions often involve unusually high or low amounts compared to typical spending behavior.
            - **Class**: The target variable (0 = legitimate, 1 = fraudulent).  
              *Role*: The label used to train the model to classify transactions. Fraudulent cases (Class = 1) are rare, making this an imbalanced classification problem.
        """)

        st.subheader("How Features Predict Fraud")
        st.markdown("""
            The XGBoost model leverages these features as follows:
            - **Time and Amount**: Directly used to detect anomalies (e.g., rapid small transactions or large unexpected ones).
            - **V1 to V28**: These PCA-derived features represent complex interactions of original variables (e.g., spending habits, location, device). XGBoost identifies non-linear patterns in these features that correlate with fraud.
            - **Combined Effect**: The model learns from the interplay of all features. For instance, a high 'Amount' combined with an unusual 'Time' and specific 'V' values might signal fraud.

            The dataset's imbalance (few frauds vs. many legitimate transactions) is addressed using SMOTE to oversample the minority class (fraud), ensuring the model learns effectively.
        """)

        # Display sample data
        st.subheader("Sample Data")
        st.write(data.head())


if __name__ == "__main__":
    main()