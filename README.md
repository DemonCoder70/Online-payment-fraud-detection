📌 README — Online Payment Fraud Detection
🔍 1. Project Overview

Online transactions generate massive revenue — but also massive losses due to fraudulent activity.
The goal of this project is to build a machine learning model that can detect potentially fraudulent payments before they are processed, reducing financial loss and enabling real-time fraud prevention systems.

This project demonstrates:

Handling highly imbalanced datasets

Feature engineering for financial transactions

Training and comparing multiple ML models

Using appropriate metrics for fraud detection (AUC, recall, precision, PR curves)

Building interpretable and business-oriented insights

📂 2. Dataset

Source: Kaggle – Online Payment Fraud Detection dataset

Rows: ~630,000 transactions

Target: isFraud (0 = legitimate, 1 = fraudulent)

Fraud rate: ~0.13% (highly imbalanced)

⚠️ Challenges in this dataset

Fraud is extremely rare → accuracy is meaningless

Many features leak information if not handled carefully

Some transaction types have very different fraud likelihood

Need to detect fraud with high recall at a manageable false-positive rate

🧹 3. Data Preprocessing

Key steps in preprocessing:

✔ Missing values

Checked for missing values (dataset had none structurally, but validated)

✔ Encoding

Transaction type encoded using one-hot encoding

✔ Scaling

Continuous features scaled using StandardScaler to stabilize tree-based model splits and logistic regression behavior.

✔ Train/Validation/Test split

70/15/15 split

stratified to preserve fraud ratio

(Optionally: time-based split for real-world simulation)

✔ Handling class imbalance

Two methods tested:

Class weights

SMOTE oversampling on training set only

Class weights performed more stably and avoided synthetic overfitting.

🤖 4. Models Trained
Model	Why it was chosen
Logistic Regression	Simple baseline, interpretable coefficients
Random Forest	Handles nonlinearities, robust to imbalance
XGBoost	High performance on tabular fraud datasets
LightGBM	Extremely efficient and accurate for large datasets

Hyperparameters tuned using:

RandomizedSearchCV

PR-AUC as primary optimization target

📊 5. Evaluation Metrics

Fraud detection requires recall, precision, and PR-AUC, not accuracy.

🎯 Key metrics used:

ROC-AUC

Precision

Recall

F1-score

Precision–Recall Curve

Confusion Matrix

Recall at fixed false-positive rate (operationally relevant)

Accuracy is included only as a sanity check but is not used for decision thresholds.

🏆 6. Results Summary

Replace these placeholders with your actual numbers once you compute them.

Best model: LightGBM
Test ROC-AUC: 0.98+ expected on this dataset
PR-AUC: Significant improvement vs baseline
Fraud recall at 5% FPR: Strong operational recall

Example interpretation (replace with your own):

At threshold T = 0.78, the model catches 92% of fraud

Only 4% of all transactions are flagged for review

Expected to reduce fraud losses by >90% with minimal review cost

📈 7. Visualizations

Recommended visualizations (and included in the notebook):

ROC curve

Precision-Recall curve

Confusion matrix

Feature importance plot

Fraud probability distribution

SHAP summaries (optional but very strong for interviews)

🧠 8. Business Value

This project demonstrates how an automated fraud detection system can:

Reduce financial loss

Lower manual review load

Create real-time intervention systems

Increase trust in the platform

Enable risk-based transaction scoring

Serve as a foundation for more complex behavioral models

🛠 9. How to Run the Code
Clone the repository
git clone https://github.com/DemonCoder70/online-payment-fraud-detection.git
cd online-payment-fraud-detection

Install dependencies
pip install -r requirements.txt

Run the notebook
jupyter notebook online-payment-fraud-detection.ipynb

🚀 10. Future Improvements

Deploy model as a REST API (FastAPI / Flask)

Build a Streamlit dashboard for real-time scoring

Add cost-sensitive evaluation:
Expected Loss = P(Fraud) × Transaction Amount

Explore deep models (TabNet, autoencoders)

Improve feature engineering (velocity features, user profiling)

📬 11. Contact

If you want to discuss the project, collaborate, or need help replicating the workflow:

Your Name
GitHub: https://github.com/DemonCoder70
