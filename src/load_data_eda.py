import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures,OneHotEncoder
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier

import shap
from sklearn.feature_selection import RFE
import os

# ignore all future warnings
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)



# For docker
#data = pd.read_csv('heart_failure_clinical_records_dataset.csv')
# For local
data = pd.read_csv('/Users/andresjr/Documents/capgemini/pruebas/ARC-datascience-task11/data/heart_failure_clinical_records_dataset.csv')


def improved_eda(data):
    print("Dataset Information:")
    print(data.info())
    
    print("\nMissing Values:")
    print(data.isnull().sum())
    
    print("\nSummary Statistics:")
    print(data.describe())
    
    print("\nClass Distribution:")
    print(data['DEATH_EVENT'].value_counts(normalize=True))
    
    # Correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()
    
    # Distribution plots for numerical features
    num_features = data.select_dtypes(include=[np.number]).columns
    n_features = len(num_features)
    
    # Box plots for numerical features
    fig, axes = plt.subplots(n_features // 3 + 1, 3, figsize=(20, 5 * (n_features // 3 + 1)))
    axes = axes.flatten()
    
    for i, col in enumerate(num_features):
        sns.boxplot(data=data, x='DEATH_EVENT', y=col, ax=axes[i])
        axes[i].set_title(f'Box Plot of {col} by DEATH_EVENT')
    
    plt.tight_layout()
    plt.show()

improved_eda(data)

# Advanced EDA
def plot_feature_distributions(data):
    n_features = len(data.columns)
    fig, axes = plt.subplots(n_features // 3 + 1, 3, figsize=(20, 5 * (n_features // 3 + 1)))
    axes = axes.flatten()
    
    for i, col in enumerate(data.columns):
        sns.histplot(data=data, x=col, hue='DEATH_EVENT', kde=True, ax=axes[i])
        axes[i].set_title(f'Distribution of {col}')
    
    plt.tight_layout()
    plt.show()

plot_feature_distributions(data)
    
# Outlier detection and handling function
def detect_outliers(data, columns, method='iqr'):
    for col in columns:
        if method == 'iqr':
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(data[col]))
            lower_bound = data[col].mean() - 3 * data[col].std()
            upper_bound = data[col].mean() + 3 * data[col].std()
        
        print(f"Outliers in {col}:")
        print(data[(data[col] < lower_bound) | (data[col] > upper_bound)][col])
    
    return data

detect_outliers(data, data.select_dtypes(include=[np.number]).columns, method='iqr')


