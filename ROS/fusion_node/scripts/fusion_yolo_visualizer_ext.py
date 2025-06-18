#!/usr/bin/env python3
"""
===========================================================================
Nombre del archivo: fusion_yolo_visualizer_ext.py

Descripción:
    Nodo de ROS que realiza la fusión de detecciones RGB y térmicas con prioridad
    configurable por clase. Aplica homografía para proyectar bounding boxes entre
    planos de imagen. Publica resultados fusionados, guarda imágenes y genera un CSV
    con los resultados.

    La configuración de las clases prioritarias para cada cámara corresponden al dataset
    con imágenes de exterior.

Funcionalidades:
    1. Sincroniza detecciones y publicaciones de imagen térmica y RGB.
    2. Proyecta y compara bounding boxes con homografía.
    3. Fusiona detecciones basándose en IoU y clase prioritaria.
    4. Publica detecciones fusionadas y guarda imágenes anotadas y CSV por frame.

Requisitos:
    - ROS (rospy, sensor_msgs, fusion_msgs)
    - message_filters
    - OpenCV
    - NumPy
    - csv

Uso:
    - Lanza este nodo junto con los publicadores de detecciones en `/thermal_detections` 
      y `/rgb_detections`, así como las cámaras.
    - Recomendación: roslaunch fusion_node fusion_yolo_ext


Autor: Ismael González Durán  
Fecha: 2025
===========================================================================
"""

import rospy
from sensor_msgs.msg import CompressedImage
from message_filters import ApproximateTimeSynchronizer
from message_filters import Subscriber as WaitSub
import cv2
import os
import numpy as np
import csv
from fusion_msgs.msg import DetectionArray, Detection, BoundingBox as BBox
from utils_vision import encode_compressed_img, decode_compressed_img, transformar_bbox, calcular_iou, save_csv

# Variable global para contar el número de frames que se han evaluado
frame_counter = 0

# Clases que tienen prioridad en cada tipo de sensor (para el modelo exterior)
CLASES_PRIORIDAD_TERMICA = ["person"]
CLASES_PRIORIDAD_RGB = ["car"]

# Matriz de homografía de térmica a RGB y su inversa
H = np.array([
    [7.64215087e-01, -2.69001316e-02, 6.03922905e+01],
    [5.29528562e-03,  7.37333274e-01, -1.07828857e+01],
    [3.25468646e-05, -6.22497906e-05,  1.00000000e+00]
], dtype=np.float64)
H_inv = np.linalg.inv(H)

# Funciones auxiliares
def clase_prioriza_termica(class_id):
    return class_id in CLASES_PRIORIDAD_TERMICA

def clase_prioriza_rgb(class_id):
    return class_id in CLASES_PRIORIDAD_RGB

# ================================
# Callback de fusión de detecciones
# ================================
def detections_callback(det_thermal_array, det_webcam_array):
    global frame_counter, pub_fusion_detections, folder_rgb, folder_thermal

    # Obtener imágenes en formato OpenCV
    img_rgb = decode_compressed_img(det_webcam_array.image)
    img_th = decode_compressed_img(det_thermal_array.image)
    if img_rgb is None or img_th is None:
        rospy.logwarn("No se pudo decodificar alguna imagen")
        return
    
    # Creación de un nuevo mensaje de detección fusionada
    fusionadas = DetectionArray()
    fusionadas.header = det_webcam_array.header
    usadas_thermal = set()  # lista de índices de detecciones térmicas ya emparejadas

    # Emparejamiento RGB-térmica (recorremos detecciones RGB para emparejar con térmica)
    for det_rgb in det_webcam_array.detections:
        best_iou = 0
        best_idx = -1
        best_det_t = None

        # Bouding box (como tupla) y clase de la detección RGB 
        bbox_rgb = (int(det_rgb.bbox.xmin), int(det_rgb.bbox.ymin),
                    int(det_rgb.bbox.xmax), int(det_rgb.bbox.ymax))
        class_id = det_rgb.class_id

        # Recorremos detecciones térmicas para buscar emparejamiento con la detección RGB
        for j, det_t in enumerate(det_thermal_array.detections):
            
            # Si la detección es de otra clase o ya se emparejó, se salta
            if det_t.class_id != class_id or j in usadas_thermal:
                continue
            
            # Calculamos la bounding box proyectada de la detección térmica y su IoU
            bbox_t_proj = transformar_bbox(det_t.bbox, H)
            iou = calcular_iou(bbox_rgb, bbox_t_proj)
            
            # Si la IoU de esta detección es mejor que la anterior, se guarda como mejor candidata para emparejar
            if iou > best_iou:
                best_iou = iou
                best_idx = j
                best_det_t = det_t

        # Solo consideramos el emparejamiento si el solapamiento (IoU) es significativo (> 0.3)
        if best_iou > 0.3 and best_det_t:
            usadas_thermal.add(best_idx)    # Marcamos la detección térmica como utilizada
            conf_rgb, conf_t = det_rgb.confidence, best_det_t.confidence  # Obtenemos la confianza de cada modelo para la detección emparejada  

            # ESTRATEGIA DE FUSIÓN SEGÚN CLASE
            # Si la clase es prioritaria para térmica (70% conf_term + 30% conf_rgb)
            if clase_prioriza_termica(class_id):
                x1, y1, x2, y2 = transformar_bbox(best_det_t.bbox, H)
                conf = 0.7 * conf_t + 0.3 * conf_rgb
            
            # Si la clase es prioritaria para RGB (50% conf_term + 50% conf_rgb)
            elif clase_prioriza_rgb(class_id):
                x1, y1, x2, y2 = bbox_rgb
                conf = 0.5 * conf_rgb + 0.5 * conf_t
            
            # Si la clase no tuviera prioridad, se unen los bounding boxes y se toma la mayor confianza
            else:
                x1_proj, y1_proj, x2_proj, y2_proj = transformar_bbox(best_det_t.bbox, H)
                x1 = min(bbox_rgb[0], x1_proj)
                y1 = min(bbox_rgb[1], y1_proj)
                x2 = max(bbox_rgb[2], x2_proj)
                y2 = max(bbox_rgb[3], y2_proj)
                conf = max(conf_rgb, conf_t)

            # Si la confianza resultante es muy baja (menor que el 40%), se descarta la detección para evitar falsos positivos
            if conf < 0.4:
                continue

            # Creamos y agregamos la nueva detección fusionada
            bbox = BBox(xmin=x1, ymin=y1, xmax=x2, ymax=y2)
            det = Detection(class_id=class_id, confidence=conf, bbox=bbox)
            fusionadas.detections.append(det)

            # Dibujamos una bounding box morada en la imagen RGB
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(img_rgb, f"{class_id} ({conf:.2f})", (x1, max(y1 - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            
            # Dibujamos la bouding boxx morada en la imagen térmica (para ello, hacemos la inversa de la proyección)
            x1_th, y1_th, x2_th, y2_th = transformar_bbox(bbox, H_inv)
            cv2.rectangle(img_th, (x1_th, y1_th), (x2_th, y2_th), (255, 0, 255), 2)
            cv2.putText(img_th, f"{class_id} ({conf:.2f})", (x1_th, max(y1_th - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    # DETECCIONES QUE NO SE PUDIERON EMPAREJAR
    # Añadir detecciones térmicas restantes (no emparejadas)
    for j, det_t in enumerate(det_thermal_array.detections):
        if j in usadas_thermal:
            continue
        
        # Aplicamos la penalización de confianza y la descartamos si es muy baja (para evitar falsos positivos)
        conf = det_t.confidence * 0.8
        if conf < 0.4:
            continue
        
        # Guardamos la detección y la dibujamos en la imagen, pero lo hacemos de color rojo (en lugar de morado)
        x1, y1, x2, y2 = transformar_bbox(det_t.bbox, H)
        bbox = BBox(xmin=x1, ymin=y1, xmax=x2, ymax=y2)
        det = Detection(class_id=det_t.class_id, confidence=conf, bbox=bbox)
        fusionadas.detections.append(det)
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img_rgb, f"{det.class_id} ({conf:.2f})", (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Añadir detecciones RGB restantes (no emparejadas)
    for det_rgb in det_webcam_array.detections:
        if not any(det_rgb.class_id == d.class_id for d in fusionadas.detections):
            
            # Aplicamos la penalización de confianza y la descartamos si es muy baja (para evitar falsos positivos)
            conf = det_rgb.confidence * 0.8
            if conf < 0.4:
                continue
            
            # Guardamos la detección y la dibujamos en la imagen, pero lo hacemos de color verde (en lugar de morado)
            x1, y1 = int(det_rgb.bbox.xmin), int(det_rgb.bbox.ymin)
            x2, y2 = int(det_rgb.bbox.xmax), int(det_rgb.bbox.ymax)
            bbox = BBox(xmin=x1, ymin=y1, xmax=x2, ymax=y2)
            det = Detection(class_id=det_rgb.class_id, confidence=conf, bbox=bbox)
            fusionadas.detections.append(det)
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_rgb, f"{det.class_id} ({conf:.2f})", (x1, max(y1 - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # PUBLICACIÓN Y GUARDADO
    # Publicamos en un tópico ROS el resultado de la fusión de las cámaras
    pub_fusion_detections.publish(fusionadas)

    # Guardamos las imágenes con las anotaciones y las detecciones en un CSV
    cv2.imwrite(os.path.join(folder_rgb, f"fusion_rgb_{frame_counter}.png"), img_rgb)
    cv2.imwrite(os.path.join(folder_thermal, f"fusion_thermal_{frame_counter}.png"), img_th)
    save_csv(fusionadas.detections, os.path.join(folder_rgb, f"fusion_{frame_counter}.csv"))
    frame_counter += 1
    rospy.loginfo(f"Guardadas imágenes y CSV fusionados {frame_counter}")

# Callback de sincronización de imágenes
def image_sync_callback(img_thermal_msg, img_rgb_msg):
    global pub_thermal_sync, pub_rgb_sync
    pub_thermal_sync.publish(img_thermal_msg)
    pub_rgb_sync.publish(img_rgb_msg)
    rospy.loginfo("Publicado imágenes sincronizadas en topics coincidencia")

# ====================
# Inicialización nodo
def main():
    global folder_rgb, folder_thermal, pub_thermal_sync, pub_rgb_sync, pub_fusion_detections

    folder_rgb = os.path.expanduser("~/Desktop/rgb_images")
    folder_thermal = os.path.expanduser("~/Desktop/thermal_images")
    os.makedirs(folder_rgb, exist_ok=True)
    os.makedirs(folder_thermal, exist_ok=True)

    rospy.init_node("sync_and_process_node")

    # Subscripción a imágenes de las cámaras y sincronización usando ATS
    sub_thermal = WaitSub("/flir_boson/image_raw/compressed", CompressedImage)
    sub_rgb = WaitSub("/usb_cam1/image_raw/compressed", CompressedImage)
    ats_images = ApproximateTimeSynchronizer([sub_thermal, sub_rgb], queue_size=30, slop=0.1)
    ats_images.registerCallback(image_sync_callback)

    # Publicadores de imágenes sincronizadas a los topics donde las reciben los modelos YOLO
    pub_thermal_sync = rospy.Publisher("/term_img_coincidence", CompressedImage, queue_size=10)
    pub_rgb_sync = rospy.Publisher("/rgb_img_coincidence", CompressedImage, queue_size=10)
    pub_fusion_detections = rospy.Publisher("/detecciones_fusionadas", DetectionArray, queue_size=10)

    # Subscripción a los topics con detecciones y sincronización usando ATS
    sub_det_thermal = WaitSub("/thermal_detections", DetectionArray)
    sub_det_webcam = WaitSub("/rgb_detections", DetectionArray)
    ats_detections = ApproximateTimeSynchronizer([sub_det_thermal, sub_det_webcam], queue_size=100, slop=0.5)
    ats_detections.registerCallback(detections_callback)

    rospy.spin()

if __name__ == "__main__":
    main()
