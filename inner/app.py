from flask import Flask, request, Response, redirect, render_template, url_for
import pickle
from keras.models import load_model, Sequential, model_from_json

def create_app():

    APP = Flask(__name__)
    json = open('sequential2.json', 'r')
    loaded_json = json.read()
    json.close()
    loaded_model = model_from_json(loaded_json)
    loaded_model.load_weights('sequential2_weights.h5')

    @APP.route('/')
    def root():
        return render_template('base.html')

    @APP.route('/health_checkpoint/', methods=['GET', 'POST'])
    def health_checkpoint():
        """function allows for input which assembles a
           vector or tensor which the Deep Neural Network
           stored in, 'loaded_model', will predict the
           likelihood of heart disease for"""
        
        if request.method == 'POST':
            bmi = request.form.get('search')
            smoking = request.form.get('search2')
            stroke = request.form.get('search3')
            age = request.form.get('search4')
            diabetic = request.form.get('search5')
            alcohol = request.form.get('search5.0')
            active = request.form.get('search6')
            sleep = request.form.get('search7')         
            v = [bmi, age, sleep]
            y_n = [smoking, stroke, diabetic, alcohol, active]
            if all(b.isdigit() for b in v):

                if all('no' in x.lower() or 'yes' in x.lower() for x in y_n):
                    y_2 = [0 if 'no' in c.lower() else 1 for c in y_n]
                    final = list(map(int, [v[0], y_2[0], y_2[1], v[1], y_2[2], y_2[3], y_2[4], v[2]]))
                    pred = process_and_predict(final)
                    return render_template('results.html', answer = str(round(pred[1] * 100, 2))+'%')

            return redirect(url_for('health_checkpoint'))
        return render_template('base.html')
    
    def process_and_predict(vect):
        return loaded_model.predict([vect])[0]

    return APP
