#!/usr/bin/env python3
"""
===========================================================================
Nombre del archivo: depth_image_saver.py

Descripción:
    Nodo de ROS que escucha imágenes de profundidad comprimidas (formato JPEG
    o PNG) y guarda tanto los datos crudos en formato `.npy` como una versión
    normalizada en `.png`. Ideal para depuración de sensores tridimensionales.

Funcionalidades:
    1. Suscripción a un tópico de imágenes de profundidad comprimidas.
    2. Conversión del mensaje ROS a imagen NumPy.
    3. Guardado de la imagen en formato `.npy` con datos reales.
    4. Guardado de la imagen como una imagen `.png` interpretable visualmente.
    5. Archivos nombrados por timestamp para evitar colisiones.

Requisitos:
    - Python 3.8 o superior
    - ROS (rospy, sensor_msgs)
    - OpenCV
    - NumPy

Uso:
    - Publica imágenes comprimidas en el tópico adecuado.
    - Lanza este nodo y los archivos se guardarán en: ~/Desktop/depth_images/

Autor: Ismael González Durán  
Fecha: 2025
===========================================================================
"""

import rospy
from sensor_msgs.msg import CompressedImage
import numpy as np
import cv2
import os

# Directorio de salida donde se guardarán las imágenes
output_dir = os.path.expanduser("~/Desktop/depth_images")
os.makedirs(output_dir, exist_ok=True)

# Callback que se ejecuta cada vez que llega una imagen comprimida
def callback(msg):
    try:
        # Convertir el buffer comprimido a un array NumPy (imagen en bruto)
        np_arr = np.frombuffer(msg.data, np.uint8)
        depth_img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

        if depth_img is None or depth_img.size == 0:
            rospy.logwarn("Imagen ilegible o vacía.")
            return

        # Obtener timestamp en nanosegundos
        timestamp = msg.header.stamp.to_nsec()

        # Guardar los datos de profundidad reales como archivo .npy
        np.save(os.path.join(output_dir, f"depth_{timestamp}.npy"), depth_img)

        # Normalizar para visualización (entre 0 y 255) y guardar como PNG
        norm = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX)
        depth_vis = norm.astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, f"depth_{timestamp}.png"), depth_vis)

        rospy.loginfo(f"Guardadas imagen depth_{timestamp}")
    except Exception as e:
        rospy.logerr(f"Error procesando imagen de profundidad: {e}")

def main():
    rospy.init_node("depth_extractor_live")

    # Suscripción al tópico de imágenes de profundidad comprimidas
    rospy.Subscriber("/robot/front_rgbd_camera/depth/image_raw/compressed", CompressedImage, callback)

    rospy.loginfo("Esperando imágenes de profundidad...")
    rospy.spin()

if __name__ == "__main__":
    main()
