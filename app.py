from flask import Flask, jsonify,render_template
import pandas as pd
import joblib
import random
import numpy
import xgboost
app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('model_xgb1.pkl')

# Load the test CSV file
test_csv_path = 'creditfraudtest.csv'  # Ensure you provide the correct path to your CSV
df_test = pd.read_csv(test_csv_path)
df_test.drop("Unnamed: 0",axis=1,inplace=True)
# Drop unnecessary columns and apply any necessary feature engineering as you've done before
drop_cols = ['cc_num','merchant','first','last','street','zip','trans_num','unix_time','is_fraud']

df_test1 = df_test.drop(drop_cols, axis=1)
df_test1['trans_date_trans_time']=pd.to_datetime(df_test['trans_date_trans_time'])
df_test1['trans_date']=df_test1['trans_date_trans_time'].dt.strftime('%Y-%m-%d')
df_test1['trans_date']=pd.to_datetime(df_test1['trans_date'])
df_test1['dob']=pd.to_datetime(df_test1['dob'])
df_test1['age_at_trans'] = ((df_test1['trans_date'] - df_test1['dob']).dt.days // 365.25)
df_test1['lat_dist'] = abs(round(df_test1['merch_lat'] - df_test1['lat'], 3))
df_test1['long_dist'] = abs(round(df_test1['merch_long'] - df_test1['long'], 3))
df_test1['trans_month'] = pd.DatetimeIndex(df_test1['trans_date']).month
drop_cols = ['trans_date_trans_time','city','lat','long','job','dob','merch_lat','merch_long','trans_date']
df_test2 = df_test1.drop(drop_cols,axis=1)
df_test2['gender'] = df_test2['gender'].map({'M': 1, 'F': 0})
# Further feature engineering (dummy encoding, etc.)
df_test3 = pd.get_dummies(df_test2,columns=['category'],drop_first=True)

# Ensure the column order matches your model's expected input
df_test4 = df_test3.drop('state',axis=1)

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict', methods=['GET'])
def predict():
    # Randomly select a row from the test data
    random_idx = random.randint(0, len(df_test4) - 1)
    random_sample = df_test4.iloc[[random_idx]]  # This will return a DataFrame

    # Make a prediction using your model
    prediction = model.predict(random_sample)[0]  # Get the first (and only) prediction
    print(prediction)
    # Return the random sample and prediction result as a JSON response
    result = {
        'selected_data': random_sample.to_dict(orient='records')[0],  # Convert DataFrame row to dict
        'prediction': int(prediction)  # Convert prediction to integer (0 or 1)
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
