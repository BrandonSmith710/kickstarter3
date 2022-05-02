from flask import Flask, render_template, request
import pickle
import pandas as pd

def create_app():

    APP = Flask(__name__)

    @APP.route('/')
    def form():
        return render_template('base.html')
    
    @APP.route('/data/', methods=['GET', 'POST'])
    def data():

        if request.method == 'POST':
            # Get form data
            name = request.form.get('name')
            blurb = request.form.get('blurb', 'default')
            country = request.form.get('country', 'default')
            backers_count = request.form.get('backers_count', 'default')
            prediction = preprocessDataAndPredict(name, blurb, country,
                                                  backers_count)
            return render_template('data.html', prediction=prediction[0])

    def preprocessDataAndPredict(name, blurb, country, backers_count):

        test_data = (name, blurb, country, backers_count)

        test_data = np.array(test_data)
        dftest = pd.DataFrame(test_data).T
        dftest.columns = ['name', 'blurb', 'country', 'backers_count']
        model = pickle.load(
            open('kickstarter/model_knn', 'rb'))

        prediction = model.predict(dftest)
        return prediction

    return APP
