#!/usr/bin/env python3
"""
===========================================================================
Nombre del archivo: utils_vision.py

Descripción:
    Módulo con funciones comunes para el procesamiento de imágenes
    y detecciones para la fusión de detecciones de cámaras RGB y térmica
    usando ROS y OpenCV.

Funciones incluidas:
    - Conversión entre imágenes comprimidas ROS y OpenCV.
    - Proyección de bounding boxes con homografía.
    - Cálculo de IoU (Intersection over Union).
    - Guardado de detecciones en archivos CSV.
    - Función callback para mandar las imagenes al topico de imagen sincronizada

Autor: Ismael González Durán
Fecha: 2025
===========================================================================
"""

import numpy as np
import cv2
import csv
from sensor_msgs.msg import CompressedImage
from fusion_msgs.msg import DetectionArray, Detection, BoundingBox as BBox

# Convierte mensaje ROS CompressedImage a imagen OpenCV
def decode_compressed_img(msg):
    np_arr = np.frombuffer(msg.data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

# Codifica imagen OpenCV a mensaje CompressedImage
def encode_compressed_img(cv_img):
    msg = CompressedImage()
    msg.format = "jpeg"
    ret, buf = cv2.imencode('.jpg', cv_img)
    msg.data = np.array(buf).tobytes()
    return msg

# Proyecta un bounding box con una homografía al plano de la cámara RGB
def transformar_bbox_con_homografia(bbox, H, shape_thermal=None):
    # Ponemos los puntos de la bounding box térmica en una matriz para poder calcular el producto
    x1, y1 = bbox.xmin, bbox.ymin
    x2, y2 = bbox.xmax, bbox.ymax

    if shape_thermal is not None:
        h, w = shape_thermal
        x1, x2 = x1*w, x2*w
        y1, y2 = y1*h, y2*h

    puntos = np.array([
        [x1, y1, 1],
        [x2, y1, 1],
        [x2, y2, 1],
        [x1, y2, 1]
    ], dtype=np.float32).T

    # Se calcula el producto matricial con la matriz de homografía
    puntos_proyectados = H @ puntos
    # Convertir a coordenadas cartesianas para poder dibujarla
    puntos_proyectados = puntos_proyectados / puntos_proyectados[2,:]

    # Valores de X e Y de los cuatro vértices transformados
    xs = puntos_proyectados[0,:]
    ys = puntos_proyectados[1,:]

    # Escogemos los vértices superior izquierdo e inferior derecho
    x_min, y_min = int(np.floor(xs.min())), int(np.floor(ys.min()))
    x_max, y_max = int(np.ceil(xs.max())), int(np.ceil(ys.max()))
    
    return max(0, x_min), max(0, y_min), x_max, y_max

# Calcula la métrica de IoU (Intersection over Union)
def calcular_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Area de intersección 
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    if interArea == 0:
        return 0.0

    # Áreas independientes de cada caja
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # Fórmula de IoU (estado del arte)
    return interArea / float(boxAArea + boxBArea - interArea)

# Guarda detecciones en CSV
def save_csv(detections, path):
    with open(path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["class_id", "confidence", "xmin", "ymin", "xmax", "ymax"])
        for det in detections:
            bbox = det.bbox
            writer.writerow([det.class_id, f"{det.confidence:.2f}", bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax])

# Callback de sincronización de imágenes
def image_sync_callback(img_thermal_msg, img_rgb_msg):
    global pub_thermal_sync, pub_rgb_sync
    pub_thermal_sync.publish(img_thermal_msg)
    pub_rgb_sync.publish(img_rgb_msg)
    rospy.loginfo("Publicado imágenes sincronizadas en topics coincidencia")