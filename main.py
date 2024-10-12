from flask import Flask, request, jsonify
from models.bodyfat_model import BodyFatModel
from models.churn_model import ChurnModel
from models.rossman_model import RossmanModel
from models.car_model import CarModel
from models.wine_model import WineModel
from models.stroke_model import StrokeModel
from models.cirrhosis_model import CirrhosisModel
from models.hepatitis_model import HepatitisModel
from models.bitcoin_model import BitcoinModel
from models.spStock_model import SPStockModel

# Crear la instancia del modelo
model = BodyFatModel('datasets/bodyfat.csv', 'models/bodyfat_model.pkl')
churn = ChurnModel('datasets/Telco-Customer-Churn.csv', 'models/churn_model.pkl')
rossmann = RossmanModel('datasets/rossman.csv', 'models/rossman_model.pkl')
car = CarModel('datasets/car.csv', 'models/car.pkl')
wine = WineModel('datasets/winequalityN.csv', 'models/wine.pkl')
stroke = StrokeModel('datasets/healthcare-dataset-stroke-data.csv', 'models/stroke.pkl')
cirrhosis = CirrhosisModel('datasets/cirrhosis.csv', 'models/cirrhosis.pkl')
hepatitis = HepatitisModel('datasets/hepatitis.csv', 'models/hepatitis.pkl')
bitcoin = BitcoinModel('datasets/bitcoin.csv', 'models/bitcoin.pkl')
spStock = SPStockModel('datasets/spStock.csv', 'models/spStock.pkl')
app = Flask(__name__)

@app.route('/bodyfat/predict', methods=['POST'])
def predict_body_fat():
    input_data = request.json
    prediction = model.predict(input_data)
    return jsonify({
        'prediction': prediction
    })

@app.route('/churn/predict', methods=['POST'])
def predict_churn():
    input_data = request.json
    prediction = churn.predict(input_data)
    return jsonify({
        'prediction': prediction
    })

@app.route('/rossman/predict', methods=['POST'])
def predict_rossmann():
    input_data = request.json
    prediction = rossmann.predict(input_data['date'])
    return jsonify({
        'prediction': prediction
    })

@app.route('/car/predict', methods=['POST'])
def predict_car():
    input_data = request.json
    prediction = car.predict(input_data)
    return jsonify({
        'prediction': prediction
    })

@app.route('/wine/predict', methods=['POST'])
def predict_wine():
    input_data = request.json
    prediction = wine.predict(input_data)
    return jsonify({
        'prediction': prediction
    })

@app.route('/stroke/predict', methods=['POST'])
def predict_stroke():
    input_data = request.json
    prediction = stroke.predict(input_data)
    return jsonify({
        'prediction': prediction
    })

@app.route('/cirrhosis/predict', methods=['POST'])
def predict_cirrhosis():
    input_data = request.json
    prediction = cirrhosis.predict(input_data)
    return jsonify({
        'prediction': prediction
    })

@app.route('/hepatitis/predict', methods=['POST'])
def predict_hepatitis():
    input_data = request.json
    prediction = hepatitis.predict(input_data)
    return jsonify({
        'prediction': prediction
    })

@app.route('/bitcoin/predict', methods=['POST'])
def predict_bitcoin():
    input_data = request.json
    prediction = bitcoin.predict(input_data['date'])
    return jsonify({
        'prediction': prediction
    })

@app.route('/spStock/predict', methods=['POST'])
def predict_spStock():
    input_data = request.json
    prediction = spStock.predict(input_data['date'])
    return jsonify({
        'prediction': prediction
    })


if __name__ == '__main__':
    app.run(debug=True)
