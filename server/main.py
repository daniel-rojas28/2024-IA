import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from models.bodyfat import BodyFatModel
from models.churn import ChurnModel
from models.rossman import RossmanModel
from models.car import CarModel
from models.wine import WineModel
from models.stroke import StrokeModel
from models.cirrhosis import CirrhosisModel
from models.hepatitis import HepatitisModel
from models.bitcoin import BitcoinModel
from models.spStock import SPStockModel
from recon import detect_dominant_emotion
from dotenv import load_dotenv
import base64

# Cargar las variables de entorno desde el archivo .env
load_dotenv()
app = Flask(__name__)
CORS(app)

@app.route('/bodyfat/predict', methods=['POST'])
def predict_body_fat():
    input_data = request.json
    prediction = bodyfat.predict(input_data)
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

@app.route('/detect-emotion', methods=['POST'])
def detect_emotion():
    image_bytes = None
    
    # Intentar obtener la imagen como multipart/form-data
    if 'image' in request.files:
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'El archivo está vacío'}), 400

        # Leer la imagen como bytes
        image_bytes = file.read()
    
    # Intentar obtener la imagen como base64 en JSON
    elif request.is_json:
        data = request.get_json()
        if data and 'image' in data:
            # Obtener la imagen en formato base64 y eliminar el prefijo
            image_base64 = data['image'].split(',')[1]

            # Decodificar la imagen de base64 a bytes
            try:
                image_bytes = base64.b64decode(image_base64)
            except Exception as e:
                return jsonify({'error': 'Error al decodificar la imagen'}), 400
    
    # Intentar obtener la imagen como base64 en request.form
    elif 'image' in request.form:
        # Obtener la imagen en formato base64 y eliminar el prefijo
        image_base64 = request.form['image'].split(',')[1]

        # Decodificar la imagen de base64 a bytes
        try:
            image_bytes = base64.b64decode(image_base64)
        except Exception as e:
            return jsonify({'error': 'Error al decodificar la imagen'}), 400

    else:
        return jsonify({'error': 'Formato de solicitud no soportado'}), 400

    # Detectar la emoción predominante
    dominant_emotion = detect_dominant_emotion(image_bytes)

    return jsonify({'dominant_emotion': dominant_emotion})
if __name__ == "__main__":
    if not os.path.isdir('ml'):
        os.mkdir(os.path.join(os.path.dirname(__file__), 'ml'))    

        # Crear la instancia del modelo
    bodyfat = BodyFatModel('datasets/bodyfat.csv')
    churn = ChurnModel('datasets/churn.csv')
    rossmann = RossmanModel('datasets/rossman.csv')
    car = CarModel('datasets/car.csv')
    wine = WineModel('datasets/wine.csv')
    stroke = StrokeModel('datasets/stroke.csv')
    cirrhosis = CirrhosisModel('datasets/cirrhosis.csv')
    hepatitis = HepatitisModel('datasets/hepatitis.csv')
    bitcoin = BitcoinModel('datasets/bitcoin.csv')
    spStock = SPStockModel('datasets/spStock.csv')
    app.run(debug=True, port=8000)
