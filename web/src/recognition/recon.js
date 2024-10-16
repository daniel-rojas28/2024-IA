
// Cargar AWS SDK
import AWS from 'aws-sdk';

// Cargar el módulo de FileSystem para leer archivos locales
import fs from 'fs';

// Configurar AWS Rekognition con las credenciales del archivo .env
AWS.config.update({
  accessKeyId: process.env.AWS_ACCESS_KEY_ID,
  secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY,
  region: process.env.AWS_DEFAULT_REGION
});

// Crear una instancia de AWS Rekognition
const rekognition = new AWS.Rekognition();

export const detectDominantEmotionFromLocalImage = async (imagePath) => {
  try {
    // Leer la imagen local y convertirla en un buffer
    const imageBytes = fs.readFileSync(imagePath);

    // Parámetros para la API de Rekognition
    const params = {
      Image: {
        Bytes: imageBytes
      },
      Attributes: ['ALL']
    };

    // Llamar a Rekognition para detectar emociones
    const data = await rekognition.detectFaces(params).promise();
    
    let dominantEmotion = null;
    let maxConfidence = 0;

    // Buscar la emoción con la confianza más alta
    data.FaceDetails.forEach((faceDetail) => {
      faceDetail.Emotions.forEach((emotion) => {
        if (emotion.Confidence > maxConfidence) {
          maxConfidence = emotion.Confidence;
          dominantEmotion = emotion.Type;
        }
      });
    });

    return dominantEmotion ? { "status": dominantEmotion } : 'No se pudo detectar ninguna emoción';
  } catch (err) {
    console.error("Error al detectar emociones:", err);
    throw err;
  }
};

// Ejemplo de uso:
// detectDominantEmotionFromLocalImage('img/test.jpg')
//   .then(dominantEmotion => console.log("Status:", dominantEmotion))
//   .catch(err => console.error(err));