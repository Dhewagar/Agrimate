from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import sklearn
import pickle

model = pickle.load(open('model.pkl','rb'))
sc = pickle.load(open('standscaler.pkl','rb'))
ms = pickle.load(open('minmaxscaler.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def fakehome():
    return render_template("fakehome.html")

@app.route('/login')
def login():
    return render_template("login.html")

@app.route('/auth')
def auth():
    return render_template("auth.html")

@app.route('/home')
def home():
    return render_template("home.html")

@app.route('/index')
def index():
    return render_template("index.html")

@app.route('/login')
def logon():
    return render_template("login.html")

@app.route('/logout')
def logout():
    return render_template("fakehome.html")

@app.route('/weather')
def weather():
    return render_template("weather.html")


@app.route("/predict", methods=['POST'])
def predict():
    N = int(request.form['Nitrogen'])
    P = int(request.form['Phosporus'])
    K = int(request.form['Potassium'])
    temp = float(request.form['Temperature'])
    humidity = float(request.form['Humidity'])
    ph = float(request.form['Ph'])
    rainfall = float(request.form['Rainfall'])

    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    scaled_features = ms.transform(single_pred)
    final_features = sc.transform(scaled_features)
    prediction = model.predict(final_features)

    crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = "{} is the best crop to be cultivated right there".format(crop)
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."

    return redirect(url_for('result', crop=result))

@app.route("/result")
def result():
    crop = request.args.get('crop', 'No crop specified')
    return render_template('result.html', crop=crop)

if __name__ == "__main__":
    app.run(debug=True)
