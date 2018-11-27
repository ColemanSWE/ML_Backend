import flask
from flask import request
import pandas as pd
from sklearn import linear_model

app = flask.Flask(__name__)

# Opening the new data frame and making a label/features.
df = pd.read_csv('facebook-fact-check.csv')
label = df['Rating']
features = df.drop('Rating', axis=1)

# Fitting our model using simple linear regression.
regression = linear_model.LinearRegression()
regression.fit(features, label)


@app.route('/predict', methods=['POST'])
def predict():
    features = request.get_json()['features']

    prediction = regression.predict(features).tolist()

    sentence = 'This post is most likely {}'
    keyword = ''

    if prediction[0] > 0:
        keyword = 'mostly true.'
    elif prediction[0] < (-1):
        keyword = 'fake news!'
    else:
        keyword = 'non-informational'

    return flask.jsonify(sentence.format(keyword))



if __name__ == '__main__':
    app.run(host='localhost', debug=True)
