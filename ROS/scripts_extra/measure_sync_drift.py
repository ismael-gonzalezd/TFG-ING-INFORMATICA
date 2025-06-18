#!/usr/bin/env python3
"""
===========================================================================
Nombre del archivo: measure_sync_drift.py

Descripción:
    Nodo de ROS para medir el desfase temporal entre imágenes y 
    detecciones provenientes de dos sensores distintos: una cámara RGB 
    y una cámara térmica. El nodo guarda un gráfico visual con los 
    desfases de sincronización cada 10 segundos.

Funcionalidades:
    1. Suscripción a imágenes RGB y térmicas comprimidas.
    2. Suscripción a las detecciones de ambas fuentes.
    3. Cálculo de la diferencia temporal entre las publicaciones de las cámaras y de las detecciones.
    4. Generación y guardado periódico de gráficos de desfase temporal.

Requisitos:
    - Python 3.8 o superior
    - ROS (rospy, sensor_msgs, fusion_msgs)
    - matplotlib
    - threading

Uso:
    - Ejecuta este nodo mientras se publican imágenes y detecciones.
    - Cada 10 segundos se guardará un gráfico en: /tmp/sync_drift_plot.png

Autor: Ismael González Durán  
Fecha: 2025
===========================================================================
"""

import rospy
from sensor_msgs.msg import CompressedImage
from fusion_msgs.msg import DetectionArray
from collections import deque
import matplotlib.pyplot as plt
import threading

# Máximo de entradas a almacenar en memoria
MAX_ENTRIES = 1000
# Umbral para sincronización temporal
SYNC_SLACK = 0.25

# Buffers para guardar los timestamps de cada tipo de mensaje
timestamps_rgb = deque(maxlen=MAX_ENTRIES)
timestamps_thermal = deque(maxlen=MAX_ENTRIES)
timestamps_rgb_det = deque(maxlen=MAX_ENTRIES)
timestamps_thermal_det = deque(maxlen=MAX_ENTRIES)

# Diferencias temporales ya calculadas
deltas_img = []
deltas_det = []

# Lock para evitar conflictos al acceder a estructuras compartidas
lock = threading.Lock()

# Callbacks para almacenar los timestamps de imágenes RGB y térmicas
def rgb_callback(msg):
    with lock:
        timestamps_rgb.append(msg.header.stamp.to_sec())

def thermal_callback(msg):
    with lock:
        timestamps_thermal.append(msg.header.stamp.to_sec())

# Callbacks para almacenar los timestamps de detecciones RGB y térmicas
def rgb_det_callback(msg):
    with lock:
        timestamps_rgb_det.append(msg.header.stamp.to_sec())

def thermal_det_callback(msg):
    with lock:
        timestamps_thermal_det.append(msg.header.stamp.to_sec())

# Función que se ejecuta periódicamente para calcular y graficar los desfases
def timer_callback(event):
    with lock:
        # Limpiar listas de diferencias temporales anteriores
        deltas_img.clear()
        deltas_det.clear()

        # Calcular diferencias entre timestamps de imágenes más cercanas
        for t_rgb in timestamps_rgb:
            if timestamps_thermal:
                closest_th = min(timestamps_thermal, key=lambda t: abs(t - t_rgb))
                deltas_img.append(abs(t_rgb - closest_th))

        # Calcular diferencias entre timestamps de detecciones más cercanas
        for t_rgb in timestamps_rgb_det:
            if timestamps_thermal_det:
                closest_th = min(timestamps_thermal_det, key=lambda t: abs(t - t_rgb))
                deltas_det.append(abs(t_rgb - closest_th))

        # Crear gráfico de desfase temporal
        plt.figure(figsize=(12, 5))

        # Desfase de imágenes
        plt.subplot(2, 1, 1)
        plt.plot(deltas_img, label='|Δt| RGB vs Thermal Images', color='blue')
        plt.axhline(SYNC_SLACK, color='red', linestyle='--', label='Slop threshold')
        plt.xlabel("Emparejamientos")
        plt.ylabel("Diferencia temporal (s)")
        plt.title("Desfase entre imágenes RGB y térmicas")
        plt.legend()

        # Desfase de detecciones
        plt.subplot(2, 1, 2)
        plt.plot(deltas_det, label='|Δt| RGB vs Thermal Detections', color='green')
        plt.axhline(SYNC_SLACK, color='red', linestyle='--')
        plt.xlabel("Emparejamientos")
        plt.ylabel("Diferencia temporal (s)")
        plt.title("Desfase entre predicciones RGB y térmicas")
        plt.legend()

        # Guardar gráfico
        plt.tight_layout()
        plt.savefig("/tmp/sync_drift_plot.png")
        plt.close()
        rospy.loginfo("[SYNC DRIFT] Gráfico actualizado en /tmp/sync_drift_plot.png")

def main():
    rospy.init_node("measure_sync_drift_node")

    # Subscripciones a imágenes comprimidas
    rospy.Subscriber("/usb_cam1/image_raw/compressed", CompressedImage, rgb_callback)
    rospy.Subscriber("/flir_boson/image_raw/compressed", CompressedImage, thermal_callback)

    # Subscripciones a detecciones
    rospy.Subscriber("/rgb_detections", DetectionArray, rgb_det_callback)
    rospy.Subscriber("/thermal_detections", DetectionArray, thermal_det_callback)

    # Llamada periódica al cálculo del gráfico (cada 10 segundos)
    rospy.Timer(rospy.Duration(10.0), timer_callback)

    rospy.loginfo("[SYNC DRIFT] Midiendo desfases de sincronización de imágenes y detecciones...")
    rospy.spin()

if __name__ == "__main__":
    main()
