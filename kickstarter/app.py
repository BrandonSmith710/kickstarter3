from flask import Flask, render_template, request
import pickle
import pandas as pd

def create_app():
    # initializes our app
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

            # print(prediction[0])
            return render_template('data.html', prediction=prediction[0])

    def preprocessDataAndPredict(name, blurb, country, backers_count):

        # test_data = (blurb)
        test_data = (name, blurb, country, backers_count)
        # print(test_data)

        test_data = np.array(test_data)
        dftest = pd.DataFrame(test_data).T
        dftest.columns = ['name', 'blurb', 'country', 'backers_count']
        print(dftest)
        print(dftest.shape)

        # test_data = test_data.reshape(1, -1)
        # print(test_data)

        #file = open("model.pkl", "wb")
        model = pickle.load(
            open('kickstarter/model_knn', 'rb'))
        # model = pickle.load(
        #     open('Kickstarter2/kickstarter/kick_model(1)', 'rb'))

        prediction = model.predict(dftest)

        # print(prediction)

        return prediction

        # return prediction
    return APP