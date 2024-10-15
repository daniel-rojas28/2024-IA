import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import Circle from './Circle';
import Background from './Background';
import Spinner from './Spinner'; // Asegúrate de tener este componente
import './index.css';
import { 
  bodyfatBody, churnBody, rossmanBody, bitcoinBody, spStockBody, 
  carBody, hepatitisBody, wineBody, strokeBody, cirrhosisBody 
} from './requests';
import Camera from './Camera'; // Asegúrate de importar tu componente de cámara

function App() {
  const [transcript, setTranscript] = useState('Usa el botón o la tecla espacio para activar el micrófono...!!');
  const [isActive, setIsActive] = useState(false);
  const [notification, setNotification] = useState('');
  const [isListening, setIsListening] = useState(false);
  const [awaitingKeyword, setAwaitingKeyword] = useState(true);
  const [showCommands, setShowCommands] = useState(false);
  const [loading, setLoading] = useState(false); 
  const recognitionRef = useRef(null);
  const [photo, setPhoto] = useState(null); // Para almacenar la foto tomada/subida

  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  const recognition = recognitionRef.current || new SpeechRecognition();
  recognitionRef.current = recognition;
  recognition.lang = 'es-ES';
  recognition.interimResults = false;
  recognition.continuous = true;

  const apiBaseUrl = 'http://localhost:8000'; // URL base del backend

  const funciones = {
    'ventas': () => callApi('/rossman/predict', rossmanBody),
    'cirrosis': () => callApi('/cirrosis/predict', cirrhosisBody),
    'masa corporal': () => callApi('/bodyfat/predict', bodyfatBody),
    'hepatitis': () => callApi('/hepatitis/predict', hepatitisBody),
    'cerebrovascular': () => callApi('/stroke/predict', strokeBody),
    'compañía celular': () => callApi('/churn/predict', churnBody),
    'vino': () => callApi('/wine/predict', wineBody),
    'acciones': () => callApi('/spStock/predict'),
    'automóvil': () => callApi('/car/predict', carBody),
    'bitcoin': () => callApi('/bitcoin/predict', bitcoinBody),
  };

  const callApi = async (endpoint, data) => {
    setLoading(true); 

    try {
      const response = await axios.post(`${apiBaseUrl}${endpoint}`, data);
      setNotification(`Resultado: ${JSON.stringify(response.data)}`);
    } catch (error) {
      setNotification('Error al ejecutar la función.');
      console.error('Error al llamar a la API:', error);
    } finally {
      setLoading(false); 
      resetState();
      setTranscript('Diga "Efrén" para empezar a hablar...!!');
    }
  };

  const handleActivation = () => {
    if (isActive) {
      recognition.stop();
      setIsActive(false);
      setTranscript('Usa el botón o la tecla espacio para activar el micrófono...!!');
      resetState();
    } else {
      recognition.start();
      setIsActive(true);
      setTranscript('Diga "Efrén" para empezar a hablar...!!');
    }
  };

  const resetState = () => {
    setIsListening(false);
    setAwaitingKeyword(true);
  };

  const toggleCommands = () => setShowCommands(!showCommands);

  useEffect(() => {
    const handleKeyPress = (event) => {
      if (event.code === 'Space') {
        event.preventDefault();
        handleActivation();
      }
    };
    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [isActive]);

  recognition.onresult = (event) => {
    const result = event.results[event.results.length - 1][0].transcript.toLowerCase();
    console.log(result);

    if (awaitingKeyword && result.includes('hola')) {
      setIsListening(true);
      setAwaitingKeyword(false);
      setTranscript('Escuchando...!!');
      setNotification(''); // Limpiar notificación
    } else if (!awaitingKeyword) {
      setTranscript(result);

      const foundFunction = Object.keys(funciones).find((keyword) => result.includes(keyword));
      if (foundFunction) {
        funciones[foundFunction]();
      }
    }
  };

  recognition.onend = () => {
    if (isActive) recognition.start();
  };

  useEffect(() => {
    if (notification) {
      const timer = setTimeout(() => setNotification(''), 5000);
      return () => clearTimeout(timer);
    }
  }, [notification]);

  const handlePhotoTaken = (imageDataUrl) => {
    setPhoto(imageDataUrl);
  };

  return (
    <div className="App">
      <Background />
      <button className="button microphone-button" onClick={handleActivation}>
        {isActive ? 'Desactivar Micrófono' : 'Activar Micrófono'}
      </button>
      <button className="button commands-button" onClick={toggleCommands}>
        Comandos
      </button>
      <h1 className="title">E.F.R.E.N</h1>
      <p className="efren">Enabled Functionality for Rapid Exploration of Numbers</p>
      <div className="container">
        <Circle isActive={isListening || loading} />
        {loading ? <Spinner /> : <p id="text-output">{transcript}</p>}
        {notification && <p className="notification">Respuesta: {notification}</p>}
      </div>

      {showCommands && (
        <div className="commands-modal">
          <button className="close-button" onClick={toggleCommands}>✖</button>
          <h2>Comandos Disponibles</h2>
          <ul>
            <li>Predecir las ventas de la compañía Rossmann.</li>
            <li>Calificar el tipo de cirrosis de un paciente.</li>
            <li>Predecir la masa corporal de un paciente.</li>
            <li>Clasificar el tipo de Hepatitis de un paciente.</li>
            <li>Predecir si un paciente tendrá un accidente cerebro-vascular.</li>
            <li>Predecir si un cliente se va a pasar de compañía celular.</li>
            <li>Clasificar la calidad del vino.</li>
            <li>Predecir el precio de las acciones del mercado SP 500.</li>
            <li>Predecir el precio de un automóvil.</li>
            <li>Predecir el precio del Bitcoin.</li>
          </ul>
        </div>
      )}

      <div className="camera-wrapper">
        <Camera onPhotoTaken={handlePhotoTaken} />
      </div>

      
    </div>
  );
}

export default App;
