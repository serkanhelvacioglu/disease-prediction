import joblib
from flask import Flask
from flask import request

import sklearn

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    # newdata = "111110101010100101010001010010111"
    data = list(request.args['symptoms'])
    # return str(year)

    rows, cols = (1, len(data))
    temp_data = [[0] * cols] * rows
    for i in range(0,len(data)):
        temp_data[0][i] = int(data[i])
    filename = "disease_prediction_model.pkl"
    loaded_model = joblib.load(filename)
    result = loaded_model.predict(temp_data)
    return str(result)



    return str(temp_data)


if __name__ == "__main__":
    app.run()