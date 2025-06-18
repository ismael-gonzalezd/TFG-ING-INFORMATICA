#!/usr/bin/env python3
"""
===========================================================================
Nombre del archivo: fusion_yolo_debug.py

Descripción:
    Nodo de debugging para sistemas de fusión de detecciones. 
    Permite comprobar si las imágenes y detecciones de una cámara térmica 
    y una webcam se están recibiendo correctamente, y además, si están 
    siendo sincronizadas.

Funcionalidades:
    1. Escucha por separado imágenes y detecciones desde ambas cámaras.
    2. Muestra por consola los timestamps para verificar latencia o fallos.
    3. Sincroniza detecciones de ambas fuentes usando ApproximateTimeSynchronizer.
    4. Imprime en consola los timestamps emparejados para comprobar el alineamiento.

Requisitos:
    - Python 3.8 o superior
    - ROS (rospy, sensor_msgs, fusion_msgs)
    - message_filters
    - NumPy
Uso:
    - Lanza el nodo mientras publicas detecciones e imágenes desde ambas cámaras.
    - Útil para depurar el pipeline de sincronización y fusión de datos.

Autor: Ismael González Durán  
Fecha: 2025
===========================================================================
"""

import rospy
from sensor_msgs.msg import CompressedImage
from fusion_msgs.msg import DetectionArray
import message_filters
import numpy as np

# Callbacks individuales para verificar que llegan mensajes
def img_thermal_cb(msg):
    rospy.loginfo(f"Imagen térmica recibida con timestamp: {msg.header.stamp.to_sec()}")

def img_webcam_cb(msg):
    rospy.loginfo(f"Imagen webcam recibida con timestamp: {msg.header.stamp.to_sec()}")

def det_thermal_cb(msg):
    rospy.loginfo(f"Detecciones térmicas recibidas con timestamp: {msg.header.stamp.to_sec()}")

def det_webcam_cb(msg):
    rospy.loginfo(f"Detecciones webcam recibidas con timestamp: {msg.header.stamp.to_sec()}")

# Callback sincronizado que se ejecuta cuando las detecciones se alinean temporalmente
def callback(det_thermal_array, det_webcam_array):
    rospy.loginfo("Callback sincronizado llamado")
    rospy.loginfo(f"Timestamps: thermal_det={det_thermal_array.header.stamp.to_sec()}, webcam_det={det_webcam_array.header.stamp.to_sec()}")

def main():
    rospy.init_node("fusion_yolo_debug", anonymous=True)
    rospy.Time.now()  # Sincroniza reloj interno si es simulado

    # Subscripciones simples para mensajes de imágenes y detecciones
    rospy.Subscriber("/flir_boson/image_raw/compressed", CompressedImage, img_thermal_cb)
    rospy.Subscriber("/usb_cam1/image_raw/compressed", CompressedImage, img_webcam_cb)
    rospy.Subscriber("/thermal_detections", DetectionArray, det_thermal_cb)
    rospy.Subscriber("/rgb_detections", DetectionArray, det_webcam_cb)

    # Subscripciones sincronizadas (solo para detecciones)
    sub_det_thermal = message_filters.Subscriber("/thermal_detections", DetectionArray)
    sub_det_webcam = message_filters.Subscriber("/rgb_detections", DetectionArray)

    # Sincronizador de mensajes con margen de error (slop) muy amplio para debug
    ats = message_filters.ApproximateTimeSynchronizer(
        [sub_det_thermal, sub_det_webcam],
        queue_size=50,
        slop=5000.0  # En segundos, exagerado para asegurarse de que hace match en pruebas
    )
    ats.registerCallback(callback)

    rospy.loginfo("Debug node started, esperando mensajes...")
    rospy.spin()

if __name__ == "__main__":
    main()
