import pandas as pd


def one_hot_encoding(df,columns):
        for column in columns:
            df = pd.concat([df,pd.get_dummies(df[column],prefix=column)],axis=1)
            df.drop(column,axis=1,inplace=True)
        return df

def feature_engineering(data):
    # Add age_group categorical feature
    data['age_group'] = pd.cut(data['age'], bins=[0, 30, 45, 60, 75, 100], labels=['Young', 'Middle-aged', 'Senior', 'Elderly', 'Very Elderly'])

    data = one_hot_encoding(data,['age_group'])

    # Add anemia and diabetes interaction feature
    data['anemia_diabetes_interaction'] = data['anaemia'] * data['diabetes']
    
    # If you are high blood pressure and/or smoke diabetes and senior or elder, you are at higher risk of heart failure. Convert this to a binary feature (1 or 0)
    data['risk_factor'] = ((data['high_blood_pressure'] == 1) | (data['smoking'] == 1)) & ((data['diabetes'] == 1)) 
    data['risk_factor'] = data['risk_factor'].astype(int)

    return data

