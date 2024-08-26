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
import joblib


# ignore all future warnings
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


def search_spaces_dict_models():

    search_space_svc = [{'svc__kernel': ['linear', 'poly','rbf'],
                        'svc__gamma': ['scale'],
                        'svc__C': [0.1, 1]}]

    search_space_lr = [{'logisticregression__C': [0.1, 1],
                        'logisticregression__penalty': ['l1', 'l2']}]

    search_space_decisiontree = [{'decisiontreeclassifier__max_depth': [2, 4, 6]}]
    search_space_randomforest = [{'randomforestclassifier__n_estimators': [10,50,100,200],
                                    'randomforestclassifier__max_features': [1,5,10]}]

    search_space_gradientboosting = [{'gradientboostingclassifier__n_estimators': [10,25,50,100],
                                    'gradientboostingclassifier__max_features': [1, 2, 5,10]}]

    search_space_adaboost = [{'adaboostclassifier__n_estimators': [10,25,50,100]}]

    search_space_bagging = [{'baggingclassifier__n_estimators': [10,25,50]}]

    search_space_knn = [{'kneighborsclassifier__n_neighbors': [5, 10, 15,25],
                        'kneighborsclassifier__weights': ['uniform', 'distance']}]


    dict_models = {
        "svc": [('svc', SVC(probability=True,random_state=42)),
                search_space_svc,
                True],

        "logistic": [('logisticregression', LogisticRegression(penalty='l1', solver='liblinear',random_state=42)),
                        search_space_lr,
                        True],
        
        "decisiontree": [('decisiontreeclassifier', DecisionTreeClassifier(random_state=42)),
                        search_space_decisiontree,
                        False],
        "randomforest": [('randomforestclassifier', RandomForestClassifier(random_state=42)),
                        search_space_randomforest,
                        False],
        "gradientboosting": [('gradientboostingclassifier', GradientBoostingClassifier(random_state=42)),
                            search_space_gradientboosting,
                            False],
        "adaboost": [('adaboostclassifier', AdaBoostClassifier(random_state=42)),
                    search_space_adaboost,
                    False],
        "bagging": [('baggingclassifier', BaggingClassifier(random_state=42)),
                    search_space_bagging, 
                    False],
        "knn": [('kneighborsclassifier', KNeighborsClassifier()),
                search_space_knn,
                True],
    }

    return dict_models


def create_folder_structure():
    folders = [
        #'data',
        #'data',
        #'notebooks',
        #'src',
        'models_results',
        'models_pkl'
    ]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)

def save_model_outputs(name, model, model2, X, X_train, X_test, y_train, y_test):
    model_dir = os.path.join('models_results', name)
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model
    model_dir_pkl = os.path.join('models_pkl', name)
    os.makedirs(model_dir_pkl, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir_pkl, 'model.pkl'))
    
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {name}')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(model_dir, 'roc_curve.png'))
    plt.close()
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    avg_precision = average_precision_score(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall Curve (AP = {avg_precision:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {name}')
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(model_dir, 'precision_recall_curve.png'))
    plt.close()
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(model_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Feature Importance (for applicable models)
    if name in ["randomforest","gradientboosting","adaboost","extratrees"]:
        #analyze_feature_importance(best_model_clf_, X_train)    
        importances = model.named_steps[model2[0][0]].feature_importances_

        feature_importance = pd.DataFrame({'feature': X.columns, 'importance': importances})
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        feature_importance = feature_importance[feature_importance['importance'] > 0]

    
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance)
        plt.title('Feature Importance for ' + name)
        plt.savefig(os.path.join(model_dir, 'fi.png'))
        plt.close()
    
    
    # Classification Report
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(model_dir, 'classification_report.csv'))

create_folder_structure()

def training(dict_models, X, X_train, X_test, y_train,y_test):

    
    polynomial = False
    subset = False
    subset_features = X.columns

    results = {}
    already_scaled = False
    for name, model in dict_models.items():
        print(f"\nTraining {name}...")

        if model[2] and not already_scaled: # this means the model requires scaling
        
            standard_scaler = StandardScaler()
        
            if subset:
                #get the subset of features
                X_train = X_train[[subset_features]]
                X_test = X_test[[subset_features]] 
                
            #check which columns are binary
            binary_columns_subset = [col for col in X_train.columns if len(X_train[col].unique()) == 2]
            #use sets to get the difference between the two lists
            numeric_features_subset = list(set(X_train.columns) - set(binary_columns_subset))

            #print(numeric_features_subset)
            #print(binary_columns_subset)

            X_train_scaled = standard_scaler.fit_transform(X_train[numeric_features_subset])
            X_train = np.concatenate((X_train_scaled, X_train[binary_columns_subset]), axis=1)
            X_train = pd.DataFrame(X_train, columns=numeric_features_subset+binary_columns_subset)

            X_test_scaled = standard_scaler.transform(X_test[numeric_features_subset])
            X_test = np.concatenate((X_test_scaled, X_test[binary_columns_subset]), axis=1)

            already_scaled = True

        if polynomial:

                poly = PolynomialFeatures(degree = 2, interaction_only=False, include_bias=True)

                X_train_poly = poly.fit_transform(X_train)
                X_test_poly = poly.fit_transform(X_test)
                
                # Append the binary columns to the normalized data
                X_train = X_train_poly
                X_train = pd.DataFrame(X_train)

                X_test = X_test_poly
        
        pipe = Pipeline([model[0]])
        grid_search = GridSearchCV(pipe, model[1], cv=5, verbose=0, scoring="roc_auc", return_train_score=True, n_jobs=-1)
        best_model = grid_search.fit(X_train.values, y_train.values)
        
        # Save model outputs
        save_model_outputs(name, best_model.best_estimator_,model,X, X_train, X_test, y_train, y_test)
        
        # Evaluate on test set
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]
        
        results[name] = {
            'model': best_model.best_estimator_,
            'best_params': best_model.best_params_,
            'cv_score': best_model.best_score_,
            'test_score': roc_auc_score(y_test, y_prob)
        }

    # Save overall results
    results_df = pd.DataFrame(results).T
    results_df.to_csv('models_results/model_comparison.csv')

    # Best model analysis
    best_model_name = max(results, key=lambda x: results[x]['cv_score'])
    print(f"\nBest Model: {best_model_name}")
    print(f"Best Parameters: {results[best_model_name]['best_params']}")
    print(f"Test ROC-AUC Score: {results[best_model_name]['test_score']:.4f}")


    return results,best_model, best_model_name

# Load and display best model's outputs

def load_display_best_model(best_model):
    best_model_dir = os.path.join('models_results', best_model)
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    img = plt.imread(os.path.join(best_model_dir, 'roc_curve.png'))
    plt.imshow(img)
    plt.axis('off')
    plt.title('ROC Curve')
    plt.subplot(132)
    img = plt.imread(os.path.join(best_model_dir, 'precision_recall_curve.png'))
    plt.imshow(img)
    plt.axis('off')
    plt.title('Precision-Recall Curve')
    plt.subplot(133)
    img = plt.imread(os.path.join(best_model_dir, 'confusion_matrix.png'))
    plt.imshow(img)
    plt.axis('off')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

    if os.path.exists(os.path.join(best_model_dir, 'fi.png')):
        plt.figure(figsize=(10, 6))
        img = plt.imread(os.path.join(best_model_dir, 'fi.png'))
        plt.imshow(img)
        plt.axis('off')
        plt.title('Feature Importance')
        plt.show()

  

def cv_scores(results):

    best_model_cv = max(results, key=lambda x: results[x]['cv_score'])
    print(f"Best Model cv: {best_model_cv}")
