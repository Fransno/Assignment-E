import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocessing():

    df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")
    df = df.drop_duplicates()

    print("\n---------------- Missing Values ----------------")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])

    print("\n---------------- Potential Outliers ----------------")
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    for col in numerical_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        
        if len(outliers) > 0:
            print(f"Column '{col}': {len(outliers)} outliers")
            print(f"Range: [{lower_bound:.2f}, {upper_bound:.2f}]")
            print(outliers[[col]].head())
    

    lb = LabelEncoder()

    df = df.drop(columns=['Person ID'])
    df['Gender'] = lb.fit_transform(df['Gender'])
    df['Occupation'] = lb.fit_transform(df['Occupation'])
    df['BMI Category'] = df['BMI Category'].replace({'Normal Weight': 'Normal'})
    df['BMI Category'] = lb.fit_transform(df['BMI Category'])
    df['Sleep Disorder'] = df['Sleep Disorder'].fillna('None') # Handle missing value
    df['Sleep Disorder'] = df['Sleep Disorder'].map({'None': 0, 'Insomnia': 1, 'Sleep Apnea': 2})

    df = df.drop(columns=['Blood Pressure'])

    return df

#    df['Sleep Duration'] = (df['Sleep Duration'] * 0.6 + df['Quality of Sleep'] * 0.4) *10
#    df = df.rename(columns={'Sleep Duration': 'Sleep Score'})
#    df = df.drop(columns=['Quality of Sleep'])
#
#    df['Physical Activity Level'] = df['Physical Activity Level'] * 0.7 + df['Daily Steps'] * 0.3 / 1000
#    df = df.rename(columns={'Physical Activity Level': 'Activity Score'})
#    df = df.drop(columns=['Daily Steps'])
#
#    df[['Blood Pressure', 'Diastolic']] = df['Blood Pressure'].str.split('/', expand=True).astype(int)
#    df['Stress Level'] = (
#        df['Stress Level'] * 20 +        # 0-10 scale → 0-200
#        df['Blood Pressure'] * 0.5 +     # ~100-160 → 50-80
#        df['Diastolic'] * 0.5 +          # ~60-100 → 30-50
#        df['Heart Rate'] +               # ~60-100 → 60-100
#        df['BMI Category'] * 20          # 0-3 → 0-60
#    )
#    df = df.rename(columns={'Stress Level': 'Health Score'})
#    df = df.drop(columns=['BMI Category', 'Blood Pressure','Heart Rate', 'Diastolic'])
#
#    df['Gender'] = df['Age'] * (2 * df['Gender'] - 1)
#    df = df.rename(columns={'Gender': 'Age/Gender'})
#    df = df.drop(columns=['Age'])

