#!/usr/bin/env python
"""
===============================================================================
Nombre del archivo: video_subscriber.py

Descripción:
    Nodo ROS que se suscribe a un topic de imágenes y muestra el video en tiempo real
    usando OpenCV. Convierte mensajes ROS tipo sensor_msgs/Image a imágenes OpenCV.

Requisitos:
    - ROS (rospy)
    - OpenCV
    - cv_bridge

Autor: Ismael González Durán
Fecha: 2025
===============================================================================
"""

import rospy                              # Cliente de ROS en Python
from sensor_msgs.msg import Image         # Tipo de mensaje ROS para imágenes sin comprimir
from cv_bridge import CvBridge            # Utilidad para convertir entre imágenes ROS y OpenCV
import cv2                                # Biblioteca OpenCV para procesamiento de imágenes

# Callback que se ejecuta al recibir una imagen
def image_callback(msg):
    bridge = CvBridge()                   # Se crea un puente para convertir entre ROS y OpenCV
    
    try:
        # Conversión de mensaje ROS (sensor_msgs/Image) a imagen OpenCV en formato BGR
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
    except Exception as e:
        rospy.logerr("Error al convertir la imagen: %s", str(e))
        return

    # Muestra la imagen en una ventana de OpenCV
    cv2.imshow("Video", cv_image)
    # Necesario para que la ventana responda (1 ms de espera)
    cv2.waitKey(1)

# Función principal
def main():
    rospy.init_node('video_subscriber', anonymous=True)  # Inicializa el nodo ROS con un nombre único
    image_topic = "/flir_boson/image_raw/compressed"      # Nombre del topic del que se quiere leer imágenes

    # Se suscribe al topic especificado y se asocia la función de callback
    rospy.Subscriber(image_topic, Image, image_callback)
    rospy.spin()

    # Al salir del spin, se cierran todas las ventanas de OpenCV
    cv2.destroyAllWindows()

# Punto de entrada del script
if __name__ == '__main__':
    main()
