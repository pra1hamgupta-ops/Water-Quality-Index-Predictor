from flask import Flask,request
from flask import render_template
from keras.models import load_model
import numpy as np
import pandas as pd

app = Flask(__name__)
model = load_model('my_model.h5')

@app.route("/", methods = ['GET', 'POST'])
def hello_world():
    WQI = -1
    columns = ['pH', 'E.C. (m/cm)', 'HCO3 (mg/L)', 'Cl (mg/L)', 'NO3 (mg/L)', 'SO4 (mg/L)', 'TH (mg/L)', 'Ca (mg/L)', 'Mg (mg/L)', 'Na (mg/L)','K (mg/L)']
    if request.method == 'POST':
        arr = []
        for x in request.form.values():
            if(x != ''):
                arr.append(float(x))
            else:
                arr.append(0)
        values = pd.Series(np.array(arr)).to_frame().T
        prediction = np.array(model.predict(values))
        WQI = round(prediction[0][0],2)
    return render_template('layout.html', columns = columns, WQI = WQI)


if __name__ == '__main__':
    app.run(debug= True)