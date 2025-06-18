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

## 🧠 Modelos entrenados para imagen

Se han entrenado varios modelos YOLO especializados en la detección de personas adaptados a diferentes entornos (exterior-interior) y sensores (RGB-térmica). A continuación se describen los conjuntos de datos y escenarios empleados:

### 🔹 Modelos para interiores

- Se capturaron imágenes en el pasillo del edificio B de la Escuela Politécnica Superior (EPS), simulando tanto situaciones normales como de emergencia (personas tumbadas, sentadas trabajando o en movimiento).
- Para enriquecer el modelo térmico se incorporaron 200 fragmentos de vídeo cedidos por Rafael Domínguez Sáez, así como imágenes adicionales del dataset público **[OpenThermalPose](https://github.com/IS2AI/OpenThermalPose)**, que aportan diversidad de posturas.
- Para el modelo RGB se usaron 225 imágenes del dataset **[MPII Human Pose](http://human-pose.mpi-inf.mpg.de/)**, complementadas con 30 imágenes generadas mediante técnicas de inteligencia artificial generativa, lo que permitió cubrir situaciones no representadas en el resto del dataset.

### 🔹 Modelos para exteriores

- Se utilizaron 1658 imágenes RGB y térmicas, repartidas de forma homogénea entre los modelos para cámaras térmicas y RGB.
- La mitad de las imágenes provienen del **[FLIR ADAS Dataset](https://www.flir.com/oem/adas/dataset/)**, elegido por su compatibilidad con la cámara térmica FLIR del robot y su variedad de escenarios urbanos y de carretera bajo distintas condiciones de luz.
- El resto se obtuvo mediante una campaña de captura de datos en el aparcamiento entre los edificios B y C de la EPS, con cámaras térmica y RGB montadas en paralelo y sincronizadas mediante ROS. Las escenas incluyen colisiones simuladas, inspecciones de motor y maniobras de aparcamiento.

Todos los modelos fueron validados con métricas de precisión y visualización de resultados, disponibles en la carpeta [`MODELOS_ENTRENADOS`](./MODELOS_ENTRENADOS).

### 🔹 Modelo para nubes de puntos 3D

Para el tratamiento de datos 3D obtenidos mediante el sensor LiDAR y la cámara RGB, se utilizó un modelo **PointNet** preentrenado, proporcionado por el repositorio [`learning3d`](https://github.com/vinits5/learning3d).


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
