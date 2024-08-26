# Heart Failure Prediction Project

## Overview

This project focuses on predicting heart failure events using machine learning models. It analyzes clinical data to identify patients at risk of heart failure, aiming to aid in early intervention and prevention of adverse outcomes.

## Project Structure

```
.
├── data/
│   ├── heart_failure_clinical_records_dataset.csv
├── notebooks/
│   ├── main.ipynb
├── src/
│   ├── load_data_eda.py
│   ├── feature_engineering.py
│   ├── main.py
│   └── model_training.py
├── Dockerfile
├── requirements.txt
├── README.md
```

- `data/`: Contains the dataset used for analysis.
- `notebooks/`: Jupyter notebooks for interactive analysis and visualization.
- `src/`: Source code for data processing, feature engineering, and model training.
- `Dockerfile`: Instructions for building the Docker container.
- `requirements.txt`: List of Python dependencies.

## Setup

### Prerequisites

- Docker and Git installed

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/andresruizc/ARC-TestDataScience-1.git
   cd ARC-TestDataScience-1
   ```

2. Build the Docker image:
   ```
   docker build -t heart-failure-prediction .
   ```

3. Run the Docker container:
   ```
   docker run -p 8888:8888 jupyter-heart-failure
   ```

4. Open the Jupyter Lab URL provided in the console output in your web browser.

## Usage

### Preferred Method: Jupyter Notebook

**I strongly recommend using this method because the plots and figures are easier to visualize.**

Once you have the Jupyter Lab interface open in your browser:

1. Open `main.ipynb`.
2. Run all the notebook cells to perform data analysis, feature engineering, model training, and evaluation.

### Alternative Method: Running the Python Script

If you prefer to run the entire analysis as a script:

1. Open a terminal in the Jupyter Lab interface.
2. Navigate to the directory.
3. Run the following command:
   ```
   python3 main.py
   ```
This method will execute the entire analysis pipeline and output results to the console and save them in the designated output directories. This approach doesn't provide the same level of interactivity and immediate visualization as the notebook method.

## Data

The dataset used in this project is the "Heart Failure Clinical Records" dataset. It contains 12 clinical features:

- age: age of the patient (years)
- anaemia: decrease of red blood cells or hemoglobin (boolean)
- creatinine phosphokinase (CPK): level of the CPK enzyme in the blood (mcg/L)
- diabetes: if the patient has diabetes (boolean)
- ejection fraction: percentage of blood leaving the heart at each contraction (percentage)
- high blood pressure: if the patient has hypertension (boolean)
- platelets: platelets in the blood (kiloplatelets/mL)
- sex: woman or man (binary)
- serum creatinine: level of serum creatinine in the blood (mg/dL)
- serum sodium: level of serum sodium in the blood (mEq/L)
- smoking: if the patient smokes or not (boolean)
- time: follow-up period (days)

The target variable is:
- death event: if the patient died during the follow-up period (boolean)

## Models

The project evaluates several machine learning models:

- Support Vector Classifier (SVC)
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- AdaBoost
- Bagging
- K-Nearest Neighbors

Each model is trained with hyperparameter tuning using GridSearchCV.

`All trained models are also stored in .pkl format in the folder called models_pkl`

## Evaluation

Models are evaluated using various metrics:

- ROC AUC Score
- Precision-Recall Curve
- Confusion Matrix
- Classification Report

The best-performing model is selected based on cross-validation scores and test set performance.

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your proposed changes.
