import boto3
import os
from dotenv import load_dotenv

load_dotenv()

aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
aws_region = os.getenv('AWS_REGION')
# Configurar el cliente de AWS Rekognition

rekognition = boto3.client('rekognition', 
    region_name=aws_region, 
    aws_access_key_id=aws_access_key_id, 
    aws_secret_access_key=aws_secret_access_key
)

def detect_dominant_emotion(image_bytes):
    try:
        response = rekognition.detect_faces(
            Image={'Bytes': image_bytes},
            Attributes=['ALL']
        )
        face_details = response.get('FaceDetails', [])
        
        if not face_details:
            return 'No se pudo detectar ninguna emoción'

        # Encontrar la emoción más predominante
        dominant_emotion = None
        max_confidence = 0

        for face in face_details:
            for emotion in face['Emotions']:
                if emotion['Confidence'] > max_confidence:
                    max_confidence = emotion['Confidence']
                    dominant_emotion = emotion['Type']

        return dominant_emotion if dominant_emotion else 'No se pudo detectar ninguna emoción'
    except Exception as e:
        print(f'Error al detectar emociones: {e}')
        return 'Error al procesar la imagen'
