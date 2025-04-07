from flask import Flask, render_template, request, jsonify, redirect, url_for
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Initialize Flask app
app = Flask(__name__)

# Load Housing dataset
housing_file = '/Users/pavithragunasekaran/Documents/DAB200/.DAB200_venv/sampleproject/Housing.csv'  # Replace with the correct file path
df = pd.read_csv(housing_file)

# Check the dataset columns to define features
features = ['area', 'bedrooms', 'bathrooms']  # Adjust column names based on your dataset
X = df[features]
y = df['price']  # Target variable (house price)

# Train models
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Route for the Home page

@app.route('/')
def home():
    return render_template('home.html')
@app.route('/index')
def index():
    return render_template('index.html')
@app.route('/imageclass')
def image():
    return render_template('imageclassify.html')

# Route for the Prediction Results page
@app.route('/result', methods=['POST'])
def result():
    data = request.form
    input_data = pd.DataFrame([{
        'area': float(data['area']),
        'bedrooms': float(data['bedrooms']),
        'bathrooms': float(data['bathrooms']),
    }])

    # Predict using the trained linear model
    prediction = linear_model.predict(input_data)[0]

    return render_template('result.html', prediction=round(prediction, 2))

# Route for the Visualization page (Chart)
@app.route('/visualization')
def visualization():
    # Prepare data for chart.js
    data = {
        'area': df['area'].tolist(),
        'price': df['price'].tolist()
    }
    return render_template('visualization.html', chart_data=data)

# Route for visualizing data (AJAX for chart)
@app.route('/chart-data')
def chart_data():
    data = {
        'area': df['area'].tolist(),
        'price': df['price'].tolist()
    }
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
