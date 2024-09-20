import pandas as pd

dataset_diabetes = '../data/external/Diabetes_Classification.xlsx'
pd_diabetes = pd.read_excel(dataset_diabetes)

pd_diabetes.Gender = pd_diabetes['Gender'].apply(lambda x: 1 if x == "female" else 0)
pd_diabetes.Diabetes = pd_diabetes['Diabetes'].apply(lambda x: 0 if x == 'No diabetes' else 1)

pd_diabetes = pd_diabetes.drop("Unnamed: 16", axis=1)
pd_diabetes = pd_diabetes.drop("Unnamed: 17", axis=1)

pd_diabetes = pd_diabetes.drop('Patient number', axis=1)
pd_diabetes.to_csv('../data/processed/diabetes.csv', index=False)
