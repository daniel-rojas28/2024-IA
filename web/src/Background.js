import React, { useEffect } from 'react';
import './index.css'; // Importa los estilos para el fondo

const Background = () => {
  useEffect(() => {
    // Obtiene el canvas y su contexto 2D
    const canvas = document.getElementById('background-canvas');
    const ctx = canvas.getContext('2d');

    // Ajusta el tamaño del canvas para que llene la ventana
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    let particlesArray = []; // Arreglo para almacenar las partículas
    const numberOfParticles = 100; // Número total de partículas a crear

    // Clase que representa una partícula
    class Particle {
      constructor(x, y) {
        this.x = x; // Posición X de la partícula
        this.y = y; // Posición Y de la partícula
        this.size = Math.random() * 3 + 1; // Tamaño aleatorio de la partícula
        this.speedX = Math.random() * 2 ; // Velocidad en el eje X
        this.speedY = Math.random() * 2 ; // Velocidad en el eje Y
      }

      // Actualiza la posición de la partícula
      update() {
        this.x += this.speedX; // Actualiza la posición X
        this.y += this.speedY; // Actualiza la posición Y

        // Comprueba si la partícula toca un borde y rebota
        if (this.x < 0 || this.x > canvas.width) {
          this.speedX *= -1; // Invertir dirección en el eje X
        }
        if (this.y < 0 || this.y > canvas.height) {
          this.speedY *= -1; // Invertir dirección en el eje Y
        }
      }

      // Dibuja la partícula en el canvas
      draw() {
        ctx.fillStyle = 'rgba(46, 204, 113, 1)'; // Color de la partícula
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2); // Dibuja un círculo
        ctx.fill(); // Rellena el círculo
      }
    }

    // Inicializa las partículas
    function init() {
      particlesArray = []; // Reinicia el arreglo de partículas
      for (let i = 0; i < numberOfParticles; i++) {
        const x = Math.random() * canvas.width; // Posición X aleatoria
        const y = Math.random() * canvas.height; // Posición Y aleatoria
        particlesArray.push(new Particle(x, y)); // Crea una nueva partícula y la añade al arreglo
      }
    }

    // Función de animación
    function animate() {
      ctx.clearRect(0, 0, canvas.width, canvas.height); // Limpia el canvas
      particlesArray.forEach((particle) => {
        particle.update(); // Actualiza la posición de la partícula
        particle.draw(); // Dibuja la partícula en el canvas
      });
      requestAnimationFrame(animate); // Llama a animate de nuevo para la siguiente fotograma
    }

    init(); // Inicializa las partículas
    animate(); // Comienza la animación

    // Limpia la animación al desmontar el componente
    return () => {
      cancelAnimationFrame(animate);
    };
  }, []); // El efecto se ejecuta una sola vez al montar el componente

  // Devuelve el canvas que se utilizará como fondo
  return <canvas id="background-canvas" className="background-canvas"></canvas>;
};

export default Background; // Exporta el componente para su uso en otras partes de la aplicación
