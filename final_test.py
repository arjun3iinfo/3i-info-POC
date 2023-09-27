import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

data_path = 'predictive_maintenance.csv'
data = pd.read_csv(data_path)
data = data.loc[data['UDI']<100]

data['Tool_wear'] = data['Tool_wear'].astype('float64')
data['Rotational_speed'] = data['Rotational_speed'].astype('float64')
#data['ProductID'] = data['ProductID'].apply(lambda x: x[1:])
#data['ProductID'] = pd.to_numeric(data['ProductID'])

df = data.copy()
df.drop(columns=['UDI','ProductID'], inplace=True)

features = [col for col in df.columns if df[col].dtype=='float64' or col =='Type']
num_features = [feature for feature in features if df[feature].dtype=='float64']

sc = StandardScaler()
type_dict = {'L': 0, 'M': 1, 'H': 2}
cause_dict = {'No Failure ': 0,
              'Power Failure ': 1,
              'Overstrain Failure ': 2,
              'Heat Dissipation Failure ': 3,
              'Tool Wear Failure ': 4}

df_pre = df.copy()
df_pre['Type'].replace(to_replace=type_dict, inplace=True)
df_pre['Failure_Type'].replace(to_replace=cause_dict, inplace=True)
df_pre[num_features] = sc.fit_transform(df_pre[num_features]) 

# Predict
# Load the saved model
model_from_file = joblib.load("newfile.pkl")
predictions = model_from_file.predict(df_pre[features])
print(predictions)