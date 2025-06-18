#!/usr/bin/env python3
"""
===========================================================================
Nombre del archivo: fusion_yolo_visualizer_iou.py

Descripción:
    Nodo de ROS encargado de:
    - Sincronizar imágenes RGB y térmicas.
    - Publicar dichas imágenes en nuevos tópicos de coincidencia.
    - Aplicar transformaciones homogéneas sobre las detecciones térmicas
      para alinearlas con el plano de la cámara RGB.
    - Calcular métricas de solapamiento (IoU) entre las detecciones RGB y térmicas.
    - Dibujar y guardar las detecciones en imágenes anotadas.

Funcionalidades:
    1. Sincroniza imágenes RGB y térmicas comprimidas.
    2. Proyecta las detecciones térmicas al plano RGB usando una homografía.
    3. Dibuja las detecciones de ambas cámaras.
    4. Calcula IoU entre las detecciones con clase coincidente.
    5. Guarda las imágenes anotadas en carpetas locales.

Requisitos:
    - ROS (rospy, sensor_msgs, fusion_msgs)
    - message_filters
    - OpenCV
    - NumPy

Uso:
    - Lanza este nodo junto con los nodos que publican en:
        `/flir_boson/image_raw/compressed`, `/usb_cam1/image_raw/compressed`,
        `/thermal_detections` y `/rgb_detections`. 
    - Recomendación: roslaunch fusion_node fusion_yolo_iou

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
from fusion_msgs.msg import DetectionArray
from utils_vision import encode_compressed_img, decode_compressed_img, transformar_bbox, calcular_iou

# Variable global para contar el número de frames que se han evaluado
frame_counter = 0


# Matriz de Homografía (H)
H = np.array([
    [7.64215087e-01, -2.69001316e-02, 6.03922905e+01],
    [5.29528562e-03,  7.37333274e-01, -1.07828857e+01],
    [3.25468646e-05, -6.22497906e-05,  1.00000000e+00]
], dtype=np.float64)

# Procesamiento de detecciones
def detections_callback(det_thermal_array, det_webcam_array):
    global frame_counter
    # Obtener imágenes en formato OpenCV
    img_webcam_cv = decode_compressed_img(det_webcam_array.image)
    img_thermal_cv = decode_compressed_img(det_thermal_array.image)
    if img_webcam_cv is None or img_thermal_cv is None:
        rospy.logwarn("No se pudo decodificar alguna imagen")
        return

    bboxes_rgb = []
    classes_rgb = []

    # Procesa y dibuja detecciones RGB (verde)
    for det in det_webcam_array.detections:
        # Vértices de la bounding box RGB original (sin transformaciones)
        x1, y1 = int(det.bbox.xmin), int(det.bbox.ymin)
        x2, y2 = int(det.bbox.xmax), int(det.bbox.ymax)
        bboxes_rgb.append((x1, y1, x2, y2))
        classes_rgb.append(det.class_id)

        cv2.rectangle(img_webcam_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_webcam_cv, f"{det.class_id} ({det.confidence:.2f})",
                    (x1, max(y1 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.rectangle(img_thermal_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Procesa y proyecta detecciones térmicas (rojo)
    for det in det_thermal_array.detections:
        # Vértices proyectados de la bounding box térmica sobre el plano RGB (usando homografia)
        x1, y1, x2, y2 = transformar_bbox_con_homografia(det.bbox, H)
        cv2.rectangle(img_webcam_cv, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img_webcam_cv, f"{det.class_id} ({det.confidence:.2f})",
                    (x1, max(y1 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Dibuja la detección térmica en su imagen original
        x1_t, y1_t = int(det.bbox.xmin), int(det.bbox.ymin)
        x2_t, y2_t = int(det.bbox.xmax), int(det.bbox.ymax)
        cv2.rectangle(img_thermal_cv, (x1_t, y1_t), (x2_t, y2_t), (0, 0, 255), 2)

        # Calcula IoU con detecciones RGB que tengan misma clase
        for (bbox_rgb, class_id_rgb) in zip(bboxes_rgb, classes_rgb):
            # Coinciden las clases RGB con las térmicas
            if class_id_rgb == det.class_id:
                # Si se supera el umbral del 30% de IoU consideramos que es el mismo objeto
                iou = calcular_iou((x1, y1, x2, y2), bbox_rgb)
                if iou > 0.3: 
                    # Escribimos el porcentaje de IoU en azul cerca de las cajas en la cámara RGB
                    cx = (max(x1, bbox_rgb[0]) + min(x2, bbox_rgb[2])) // 2
                    cy = (max(y1, bbox_rgb[1]) + min(y2, bbox_rgb[3])) // 2
                    cv2.putText(img_webcam_cv, f"IoU: {iou:.2f}", (cx, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # Guarda las imágenes procesadas en disco
    success = cv2.imwrite(os.path.join(folder_rgb, f"rgb_{frame_counter}.png"), img_webcam_cv)
    if not success:
        rospy.logwarn("No se pudo guardar imagen RGB.")
    success = cv2.imwrite(os.path.join(folder_thermal, f"thermal_{frame_counter}.png"), img_thermal_cv)
    frame_counter += 1

    rospy.loginfo(f"Guardadas imágenes {frame_counter}")

# Función main
def main():
    global folder_rgb, folder_thermal, pub_thermal_sync, pub_rgb_sync

    # Directorios donde se guardarán las imágenes anotadas
    folder_rgb = os.path.expanduser("~/Desktop/img_rgb_iou")
    folder_thermal = os.path.expanduser("~/Desktop/img_thermal_iou")
    os.makedirs(folder_rgb, exist_ok=True)
    os.makedirs(folder_thermal, exist_ok=True)

    rospy.init_node("sync_and_process_node")

    # Subscripción a imágenes comprimidas de las cámaras
    sub_thermal = WaitSub("/flir_boson/image_raw/compressed", CompressedImage)
    sub_rgb = WaitSub("/usb_cam1/image_raw/compressed", CompressedImage)

    # Sincronización de imágenes RGB y térmicas usando ATS
    ats_images = ApproximateTimeSynchronizer([sub_thermal, sub_rgb], queue_size=25, slop=0.1)
    ats_images.registerCallback(image_sync_callback)

    # Publicadores de imágenes sincronizadas a los topics donde las reciben los modelos YOLO
    pub_thermal_sync = rospy.Publisher("/term_img_coincidence", CompressedImage, queue_size=10)
    pub_rgb_sync = rospy.Publisher("/rgb_img_coincidence", CompressedImage, queue_size=10)

    # Subscripción a detecciones de los modelos YOLO
    sub_det_thermal = WaitSub("/thermal_detections", DetectionArray)
    sub_det_webcam = WaitSub("/rgb_detections", DetectionArray)

    # Sincronización de detecciones usando ATS
    ats_detections = ApproximateTimeSynchronizer([sub_det_thermal, sub_det_webcam], queue_size=30, slop=0.1)
    ats_detections.registerCallback(detections_callback)

    rospy.spin()

if __name__ == "__main__":
    main()
