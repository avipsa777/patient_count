## TABLE, Graph AND FORM
# Import necessary libraries
from flask import Flask, render_template, request, jsonify, send_file
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

app = Flask(__name__, static_url_path='/static')

# Load the trained model
model = joblib.load('patient_count_model.pkl')

# Define the hours for which you want to generate forecasts
future_hours = [1 ,2 ,3 ,4 ,5 ,6 ,7 ,8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

# Load your patient data and prepare it 
df = pd.read_csv('New_generated_data.csv')
df['Arrival Time'] = pd.to_datetime(df['Arrival Time'])
df['Date'] = pd.to_datetime(df['Date'])
df['Hour'] = df['Arrival Time'].dt.hour
patient_count = df.groupby(['Hour', df['Date'].dt.dayofweek]).size()
patient_count = patient_count.astype(np.int64).reset_index(name='PatientCount')

X = patient_count[['Hour']].values.reshape(-1, 1)
y = patient_count['PatientCount'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/forecast', methods=['POST'])
def forecast():
    if request.method == 'POST':
        selected_hour = int(request.form['hour'])
        forecasted_count = model.predict([[selected_hour]])[0]
        return jsonify({'forecasted_count': forecasted_count})

@app.route('/all_forecasts')
def all_forecasts():
    all_forecasts = [{'hour': hour, 'forecasted_count': model.predict([[hour]])[0]} for hour in future_hours]
    return jsonify(all_forecasts)

# New route to generate and serve the graph
@app.route('/generate_graph')
def generate_graph():
    # Create a scatter plot for actual patient counts
    y_pred = model.predict(X_test)
    plt.scatter(X_test, y_test, color='b', label='Actual')

    # Add a line plot for predicted patient counts (using future_hours)
    plt.plot(future_hours, model.predict(np.array(future_hours).reshape(-1, 1)), color='r', label='Predicted')

    # Customize the graph (labels, legend, etc.)
    plt.xlabel("Hour of the Day")
    plt.ylabel("Patient count")
    plt.legend()

    # Save the graph to a file
    plt.savefig('static/patient_count_graph.png')

    return send_file('static/patient_count_graph.png', mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)


