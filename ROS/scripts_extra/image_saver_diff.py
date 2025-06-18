#!/usr/bin/env python3
"""
===========================================================================
Nombre del archivo: image_saver_diff.py

Descripción:
    Este script permite detectar y guardar automáticamente imágenes que 
    presentan diferencias significativas respecto a la anterior, utilizando 
    el índice de similitud estructural (SSIM). Está diseñado para trabajar 
    con imágenes comprimidas recibidas desde un tópico de ROS.

Funcionalidades:
    1. Suscripción a un tópico de imágenes comprimidas.
    2. Conversión de mensajes ROS a imágenes OpenCV.
    3. Comparación de cada imagen nueva con la anterior usando SSIM.
    4. Guardar de imágenes diferentes en una carpeta local.

Requisitos:
    - Python 3.8 o superior
    - ROS (rospy)
    - sensor_msgs/CompressedImage
    - OpenCV (cv2)
    - NumPy
    - scikit-image
    - cv_bridge

Uso:
    - Lanza este nodo desde ROS (comando rosrun).
    - Suscríbete a un tópico de tipo sensor_msgs/CompressedImage.
    - Las imágenes significativamente distintas se guardarán en disco.

Autor: Ismael González Durán
Fecha: 2025
===========================================================================
"""
import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim

output_dir = "/carpeta_para_guardar_imagenes"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

class ImageDifferentiator:
    def __init__(self):
        self.bridge = CvBridge()
        self.prev_frame = None
        self.frame_count = 0

        # Suscripción al tópico de imágenes comprimidas 
        self.image_sub = rospy.Subscriber("/topic_camara", CompressedImage, self.image_callback)



    def image_callback(self, msg):
        # Convertir el mensaje de ROS a imagen OpenCV 
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if self.prev_frame is not None:
            # Conversión a escala de grises para calcular SSIM
            gray1 = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Calcular el indice SSIM
            score, _ = ssim(gray1, gray2, full=True)

            # Si el SSIM es bajo, significa que las imagenes son diferentes (y la guardamos)
            if score < 0.85 or self.frame_count == 1: 
                image_filename = os.path.join(output_dir, "image__term_IM2R:{}.jpg".format(self.frame_count))
                cv2.imwrite(image_filename, frame)

        # Actualizar el frame anterior y el contador
        self.prev_frame = frame
        self.frame_count += 1

def main():
    rospy.init_node('image_differentiator', anonymous=True)
    ImageDifferentiator()
    rospy.spin()

if __name__ == "__main__":
    main()
