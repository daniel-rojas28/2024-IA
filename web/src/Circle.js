import React from 'react';
import './index.css'; // Importamos los estilos

function Circle({ isActive, onClick }) {
  return (
    <div className={`circle-wrapper ${isActive ? 'active' : ''}`} onClick={onClick}>
      <div className="outer-ring">
        <div className="inner-ring">
          <div className="pulse"></div>
        </div>
      </div>
    </div>
  );
}

export default Circle;
