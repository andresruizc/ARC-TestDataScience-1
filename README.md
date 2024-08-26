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

## Setup

### Prerequisites

- Docker

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/andresruizc/abc-datascience.git
   cd abc-datascience
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


## Run

There are two ways to run the project. I recommend the first option since you'll be able to check model output plots/images

1. going to notebook and then run all
2. going to the console and run oython3 main.pys

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