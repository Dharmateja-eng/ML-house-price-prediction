from flask import Flask, render_template, request
import joblib
from datetime import datetime

app = Flask(__name__)

model = joblib.load("house_price_model.pkl")

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None

    if request.method == "POST":

        bedrooms = float(request.form["bedrooms"])
        bathrooms = float(request.form["bathrooms"])
        sqft_living = float(request.form["sqft_living"])
        sqft_lot = float(request.form["sqft_lot"])
        floors = float(request.form["floors"])
        waterfront = int(request.form["waterfront"])
        view = int(request.form["view"])
        condition = int(request.form["condition"])
        current_date = datetime.now()

        year = current_date.year
        month = current_date.month

        prediction = model.predict([[
            bedrooms,
            bathrooms,
            sqft_living,
            sqft_lot,
            floors,
            waterfront,
            view,
            condition,
            year,
            month
        ]])[0]

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)