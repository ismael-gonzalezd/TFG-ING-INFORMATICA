# TFG - Ingeniería Informática

Este repositorio contiene el Trabajo de Fin de Grado (TFG) titulado:  Sistema robótico inteligente para evaluación de emergencias mediante información visual

Autor: Ismael González Durán   

Tutor: Juan Jesús Roldán Gómez  

Curso académico: 2024/2025

---
## 🔍 Descripción del trabajo:
TFG en Ingeniería Informática centrado en el desarrollo de un sistema de percepción para entornos robóticos. El proyecto integra técnicas de visión por computador y aprendizaje profundo con el framework ROS, utilizando datos provenientes de cuatro tipos de sensores diferentes (RGB, RGB-D, térmico y LiDAR), 
montados sobre una plataforma robótica para búsqueda y rescate, con el objetivo de detectar objetos y alertar de peligros en situaciones complejas y cambiantes.

Esta fusión sensorial permite al sistema aprovechar las ventajas de los distintos sensores, por ejemplo, aprovechando que el sensor térmico detecta personas con mayor precisión que el RGB.
## 📁 Estructura del repositorio
├── MODELOS_ENTRENADOS # Modelos generados tras el entrenamiento, se incluye gráficas de validación para cada uno de ellos. 

├── NOTEBOOKS_ENTRENAR # Notebooks de entrenamiento de modelos YOLO

├── ROS # Código para integrar con ROS (Robot Operating System): scripts de ROS útiles para recolección de datos, mensajes personalizados, código para el sistema de fusión y de cada módulo de detección de peligros con sus respectivos lanzadores

├── SCRIPTS # Scripts auxiliares y herramientas que no se basan en ROS. Incluye, por ejemplo, un script interactivo para calcular la matriz de homografía entre una imagen térmica y otra RGB. 

## ⚙️ Tecnologías utilizadas

- Python 3.8
- Jupyter Notebooks /Google Colab
- PyTorch
- OpenCV
- NumPy
- pandas
- scikit-learn
- matplotlib
- OpenCV
- Open3D
- ROS Melodic (compatibilidad con Ubuntu 18.04)
---
