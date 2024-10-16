import './index.css';
import React, { useState, useRef, useEffect } from 'react';

function Camera({ onPhotoTaken }) {
  const [isCameraOpen, setIsCameraOpen] = useState(true);
  const [capturedPhoto, setCapturedPhoto] = useState(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  // Inicializa la cámara al cargar el componente
  useEffect(() => {
    if (isCameraOpen) {
      openCamera();
    }
  }, [isCameraOpen]);

  // Abrir la cámara
  const openCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      videoRef.current.srcObject = stream;

      // Espera a que el video esté listo antes de reproducirlo
      videoRef.current.addEventListener('loadedmetadata', () => {
        videoRef.current.play().catch((error) => {
          console.error('Error al reproducir el video:', error);
        });
      });
    } catch (error) {
      console.error('Error al acceder a la cámara:', error);
    }
  };

  // Tomar foto desde la cámara, usando el tamaño original del video
  const handleTakePhoto = () => {
    // Configura el tamaño del canvas al tamaño del video
    const videoWidth = videoRef.current.videoWidth;
    const videoHeight = videoRef.current.videoHeight;
    canvasRef.current.width = videoWidth;
    canvasRef.current.height = videoHeight;

    const context = canvasRef.current.getContext('2d');
    context.drawImage(videoRef.current, 0, 0, videoWidth, videoHeight);

    // Obtiene la imagen capturada en su tamaño original
    const photoData = canvasRef.current.toDataURL('image/jpeg');
    setCapturedPhoto(photoData);
    onPhotoTaken(photoData);
    setIsCameraOpen(false); // Cerrar la cámara después de tomar la foto
  };

  // Subir foto desde archivo
  const handlePhotoUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        const uploadedPhoto = reader.result;
        setCapturedPhoto(uploadedPhoto);
        onPhotoTaken(uploadedPhoto);
        setIsCameraOpen(false); // Cerrar la cámara si se sube una foto
      };
      reader.readAsDataURL(file);
    }
  };

  // Volver a abrir la cámara
  const handleRetake = () => {
    setCapturedPhoto(null);
    setIsCameraOpen(true);
  };

  return (
    <div className="camera-container">
      {isCameraOpen && (
        <video ref={videoRef} autoPlay className="video-stream" />
      )}
      {!isCameraOpen && capturedPhoto && (
        <img src={capturedPhoto} alt="Captured" className="captured-photo" />
      )}
      <canvas ref={canvasRef} style={{ display: 'none' }} />

      <div className="camera-buttons">
        {isCameraOpen ? (
          <button onClick={handleTakePhoto}>Tomar Foto</button>
        ) : (
          <button onClick={handleRetake}>Volver a Tomar</button>
        )}
        
        <div className="upload-button-wrapper">
          <button onClick={() => document.getElementById('fileInput').click()}>
            Cargar Foto
          </button>
          <input 
            type="file" 
            accept="image/*" 
            id="fileInput" 
            onChange={handlePhotoUpload} 
            style={{ display: 'none' }} // Oculta el input de archivo
          />
        </div>
      </div>
    </div>
  );
}

export default Camera;


