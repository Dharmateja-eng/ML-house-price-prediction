from flask import Flask, render_template, request
import joblib
from datetime import datetime

app = Flask(__name__)

model = joblib.load("house_price_model.pkl")


@app.route("/", methods=["GET", "POST"])
def home():

    prediction_usd = None
    prediction_inr = None

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

        prediction_usd = model.predict([[
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

        usd_to_inr = 87
        prediction_inr = prediction_usd * usd_to_inr

    return render_template(
        "index.html",
        prediction_usd=prediction_usd,
        prediction_inr=prediction_inr
    )


if __name__ == "__main__":
    app.run(debug=True)