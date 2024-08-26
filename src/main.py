from load_data_eda import *
from feature_engineering import *
from model_training import *


data = pd.read_csv('heart_failure_clinical_records_dataset.csv')

improved_eda(data)
plot_feature_distributions(data)
detect_outliers(data, data.select_dtypes(include=[np.number]).columns, method='iqr')

X = data.drop(['DEATH_EVENT'], axis=1)
y = data['DEATH_EVENT']

# Feature Engineering
X = feature_engineering(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Training set shape: ", X_train.shape)
print("Test set shape: ", X_test.shape)
  
dict_models = search_spaces_dict_models()

print(dict_models)

create_folder_structure()

results, best_model_clf, best_model_name = training(dict_models, X,X_train, X_test, y_train,y_test)

print(results)

load_display_best_model(best_model_name)

cv_scores(results)
