import React, { useRef, useState } from "react";
import Webcam from "react-webcam";
import axios from "axios";

const FaceLogin = ({ onLoginSuccess }) => {
  const webcamRef = useRef(null);
  const [image, setImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");

  // Configuración de la cámara
  const videoConstraints = {
    width: 480,
    height: 360,
    facingMode: "user",
  };

  // Captura la imagen desde la cámara
  const captureImage = () => {
    const imageSrc = webcamRef.current.getScreenshot();
    setImage(imageSrc);
    sendImageToApi(imageSrc);
  };

  // Enviar imagen a la API
  
  const sendImageToApi = async (imageData) => {
    setLoading(true);
    setMessage("");
  
    try {
      // Asegúrate de que la imagen sea una cadena en formato base64
      const base64Image = imageData.split(",")[1]; // Elimina el prefijo si existe
  
      const response = await axios.post(
        "https://619c-34-75-123-129.ngrok-free.app/login",
        {
          image: base64Image, // Enviar como "image"
        },
        {
          headers: {
            "Content-Type": "application/json", // Asegúrate de usar JSON
          },
        }
      );
  
      if (response.data.success) {
        setMessage("Login exitoso");
        onLoginSuccess(); // Notifica que el login fue exitoso
      } else {
        setMessage("Login fallido: " + response.data.message);
      }
    } catch (error) {
      setMessage("Error al autenticar: " + (error.response?.data?.error || error.message));
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <div style={{ textAlign: "center", marginTop: "50px" }}>
      <h1>Login con Reconocimiento Facial</h1>
      {!image && (
        <Webcam
          audio={false}
          ref={webcamRef}
          screenshotFormat="image/jpeg"
          videoConstraints={videoConstraints}
          style={{ marginBottom: "20px", border: "2px solid #ccc" }}
        />
      )}
      <div>
        {image ? (
          <img
            src={image}
            alt="Captured"
            style={{ width: "480px", height: "360px", marginBottom: "20px" }}
          />
        ) : null}
      </div>
      <button
        onClick={image ? () => setImage(null) : captureImage}
        style={{
          padding: "10px 20px",
          fontSize: "16px",
          cursor: "pointer",
          background: "#007BFF",
          color: "#fff",
          border: "none",
          borderRadius: "5px",
        }}
        disabled={loading}
      >
        {loading ? "Procesando..." : image ? "Intentar de nuevo" : "Logearse"}
      </button>
      {message && <p style={{ marginTop: "20px", color: message.includes("exitoso") ? "green" : "red" }}>{message}</p>}
    </div>
  );
};

export default FaceLogin;
