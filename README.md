# Credit Card Fraud Detection using XGBoost & Streamlit

## Overview
This project aims to detect fraudulent credit card transactions using machine learning. It utilizes the **Kaggle Credit Card Fraud Dataset**, which contains anonymized transaction features. The model is built using **XGBoost**, with handling of class imbalance using **SMOTE**. The app is deployed using **Streamlit** for an interactive user interface.

![Screenshot 2025-03-21 183448](https://github.com/user-attachments/assets/ea4068a8-ebd4-4e8d-8985-f1dd74d63bd5)



## Features
- **Real-time fraud detection** using a trained XGBoost model.
- **Interactive UI** built with Streamlit.
- **Data preprocessing** with scaling and oversampling (SMOTE) for class imbalance.
- **Model evaluation** with classification report and confusion matrix.
- **Visualizations** for model insights and fraud analysis.

![Screenshot 2025-03-21 183556](https://github.com/user-attachments/assets/2e38be7b-55f6-4090-9352-7b9f73cc7be9)

## Dataset
The dataset used is **Kaggle's Credit Card Fraud Dataset**, which consists of:
- `Time`: Seconds elapsed between the transaction and the first transaction.
- `V1` to `V28`: Principal Components derived from PCA (original features are anonymized).
- `Amount`: The transaction amount.
- `Class`: The target variable (0 = legitimate, 1 = fraudulent).

![Screenshot 2025-03-21 183239](https://github.com/user-attachments/assets/b37ea156-67e9-4008-8f78-eb0c9ee0de28)

Due to the highly imbalanced nature of the dataset (fraud cases are rare), **SMOTE** (Synthetic Minority Over-sampling Technique) is applied to balance the dataset.

## Installation
To run this project locally, follow these steps:

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Shhubhamvishwakarma05/Fraud_Detection.git
cd Fraud_Detection
```

### 2️⃣ Install dependencies
Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
Install required packages:
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit app
```bash
streamlit run app.py
```

## Project Structure
```
📂 credit-card-fraud-detection
├── 📄 app.py                  # Main Streamlit application
├── 📄 requirements.txt        # Required dependencies
├── 📄 README.md               # Documentation
├── 📂 data                    # Dataset (not included, download from Kaggle)
└── 📂 models                  # Saved trained models (optional)
```

## Model Training
The model is trained using the **XGBoost** classifier with the following steps:
1. **Load & preprocess data**
2. **Split dataset** (80% train, 20% test)
3. **Feature scaling** with `StandardScaler`
4. **SMOTE** applied to balance fraud cases
5. **Train XGBoost classifier**
6. **Evaluate model performance**


## Future Improvements
- 🚀 **Improve fraud detection accuracy** using deep learning models.
- 📊 **Enhance visualization** with interactive fraud insights.
- 🔍 **Add explainability tools** to understand model predictions.
- 🌍 **Deploy on cloud platforms** like AWS/GCP for scalability.

## Author
👤 **Shubham Vishwakarma**
- LinkedIn: [Shubham Vishwakarma](https://www.linkedin.com/in/shubhamvishwakarma05/)
- GitHub: [Shubhamvishwakarma05](https://github.com/Shubhamvishwakarma05)

---
🚀 **Let's detect fraud and make online transactions safer!** 💳


