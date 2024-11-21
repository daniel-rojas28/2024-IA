import React, { useState } from "react";
import ReactDOM from "react-dom/client";
import FaceLogin from "./Login"; // Importa el componente de Login
import App from "./App"; // Importa la aplicación principal

const Root = () => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  // Función para manejar la autenticación
  const handleLoginSuccess = () => {
    setIsAuthenticated(true);
  };

  return (
    <React.StrictMode>
      {isAuthenticated ? (
        <App /> // Si está autenticado, muestra la app principal
      ) : (
        <FaceLogin onLoginSuccess={handleLoginSuccess} /> // Muestra el login si no lo está
      )}
    </React.StrictMode>
  );
};

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(<Root />);
