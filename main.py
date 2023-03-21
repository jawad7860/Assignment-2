"""
This module implements a Flask application for predicting stock prices.
"""

import sys

try:
    from flask import Flask, render_template, request
    import joblib
except ImportError as e:
    print(e)
    sys.exit(1)

app = Flask(__name__)


@app.route('/')
def home():
    """
    Render the homepage template.
    """
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict the stock price based on the form data.
    """
    loaded_model = joblib.load('model1.pkl')

    open_price = float(request.form['open'])
    high_price = float(request.form['high'])
    low_price = float(request.form['low'])
    volume = float(request.form['volume'])

    # Make prediction
    new_features = [[open_price, high_price, low_price, volume]]
    predicted_price = loaded_model.predict(new_features)

    return render_template('index.html', prediction=f'Predicted Price: {predicted_price[0]}')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
    sys.exit()
