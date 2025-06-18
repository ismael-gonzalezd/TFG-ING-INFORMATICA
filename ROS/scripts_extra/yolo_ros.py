#!/usr/bin/env python3
"""
===========================================================================
Nombre del archivo: yolo_ros.py

Descripción:
    Nodo de ROS que realiza detección de objetos sobre imágenes comprimidas utilizando 
    un modelo YOLO personalizado. Las detecciones se visualizan en tiempo real 
    mediante una ventana de OpenCV.

Funcionalidades:
    1. Suscripción a un tópico de imágenes comprtérmicasimidas.
    2. Conversión de mensajes ROS a imágenes OpenCV.
    3. Detección de objetos con un modelo YOLO cargado desde un archivo `.pt`.
    4. Visualización de los resultados con bounding boxes y sus etiquetas de clase.

Requisitos:
    - Python 3.8 o superior
    - ROS (rospy, sensor_msgs)
    - OpenCV (cv2)
    - NumPy
    - cv_bridge
    - ultralytics (YOLOv8)

Uso:
    - Asegúrate de que el modelo está en el directorio especificado
    - Lanza este nodo en ROS mientras publicas imágenes en el tópico adecuado.
    - Las detecciones se visualizarán en una ventana emergente.

Autor: Ismael González Durán
Fecha: 2025
===========================================================================
"""

import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import numpy as np

# Inicializa el puente para convertir imágenes ROS a OpenCV
bridge = CvBridge()

# Carga el modelo YOLO entrenado para detección en imágenes térmicas
model = YOLO("ruta_modelo_yolo.pt")

def image_callback(msg):
    try:
        # Convertir el mensaje ROS a array de NumPy
        np_arr = np.frombuffer(msg.data, np.uint8)

        # Decodificar el array como imagen JPEG
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Aplicar el modelo YOLO a la imagen
        results = model(cv_image)

        # Dibujar resultados sobre la imagen si hay detecciones
        if results:
            for result in results:
                boxes = result.boxes  # Obtener todas las cajas detectadas
                for box in boxes:
                    # Extraer coordenadas del bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Formatear etiqueta con nombre de clase y la confianza
                    label = f"{model.names[int(box.cls)]} ({box.conf[0]:.2f})"

                    # Dibujar el bounding box y el texto sobre la imagen
                    cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(cv_image, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Mostrar la imagen en una ventana
        cv2.imshow("YOLO Detection", cv_image)
        cv2.waitKey(1)

    except Exception as e:
        rospy.logerr(f"Error en la detección: {e}")

if __name__ == "__main__":
    # Inicializa el nodo ROS
    rospy.init_node("mi_yolo")

    # Suscripción al tópico de imágenes comprimidas de la cámara térmica
    rospy.Subscriber("topico_ros_de_imagen", CompressedImage, image_callback)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Cerrando nodo...")
    finally:
        # Cierra todas las ventanas de OpenCV al finalizar
        cv2.destroyAllWindows()
