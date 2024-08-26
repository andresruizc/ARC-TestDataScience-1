# Heart Failure Prediction Project

## Overview

This project aims to predict the likelihood of heart failure events using machine learning techniques. We use a dataset containing various clinical features to build and evaluate multiple prediction models.

## Dataset

The dataset used in this project is the "Heart Failure Clinical Records" dataset, which contains 12 clinical features:

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
- [target] death event: if the patient died during the follow-up period (boolean)

Dataset source: [Heart Failure Clinical Records Dataset](https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records)

## Project Structure

```
.
├── data/
│   ├── raw/
│   │   └── heart_failure_clinical_records_dataset.csv
│   └── processed/
├── notebooks/
│   ├── 01_data_loading_and_preprocessing.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training_and_evaluation.ipynb
│   └── 05_results_and_conclusion.ipynb
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── model_evaluation.py
├── models/
├── results/
├── tests/
├── Dockerfile
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup

### Prerequisites

- Docker

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/heart-failure-prediction.git
   cd heart-failure-prediction
   ```

2. Build the Docker image:
   ```
   docker build -t heart-failure-prediction .
   ```

3. Run the Docker container:
   ```
   docker run -p 8888:8888 -v $(pwd):/app heart-failure-prediction
   ```

4. Open the Jupyter Lab URL provided in the console output in your web browser.

## Usage

Navigate through the notebooks in the following order:

1. `01_data_loading_and_preprocessing.ipynb`
2. `02_exploratory_data_analysis.ipynb`
3. `03_feature_engineering.ipynb`
4. `04_model_training_and_evaluation.ipynb`
5. `05_results_and_conclusion.ipynb`

Each notebook contains detailed comments and markdown cells explaining the steps and decisions made throughout the analysis.

## Models

We evaluate several machine learning models, including:

- Logistic Regression
- Support Vector Machines
- Decision Trees
- Random Forests
- Gradient Boosting
- K-Nearest Neighbors

The best performing model and its evaluation metrics can be found in the `05_results_and_conclusion.ipynb` notebook.

## Results

The detailed results, including model comparisons and feature importance, are available in the `results/` directory and the `05_results_and_conclusion.ipynb` notebook.

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- UCI Machine Learning Repository for providing the dataset
- All contributors and maintainers of the libraries used in this project