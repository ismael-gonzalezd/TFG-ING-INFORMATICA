#!/usr/bin/env python3
"""
===========================================================================
Nombre del archivo: fusion_yolo_view.py

Descripción:
    Nodo de ROS que visualiza detecciones de objetos en imágenes RGB y térmicas.
    Utiliza sincronización aproximada entre mensajes de tipo `DetectionArray`
    que incluyen imágenes y detecciones por modelos YOLO. Las imagenes se muestran lado
    a lado con bounding boxes para los objetos detectados. 

Funcionalidades:
    1. Escucha detecciones RGB y térmicas.
    2. Extrae imágenes comprimidas y las decodifica a formato OpenCV.
    3. Dibuja bounding boxes sobre ambas imágenes.
    4. Presenta una visualización conjunta en una única ventana.

Requisitos:
    - Python 3.8 o superior
    - ROS (rospy, message_filters, sensor_msgs, fusion_msgs)
    - OpenCV
    - NumPy

Uso:
    - Publica mensajes tipo `fusion_msgs/DetectionArray` en los tópicos esperados.
    - Lanza este nodo para ver los resultados en tiempo real.

Autor: Ismael González Durán  
Fecha: 2025
===========================================================================
"""

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage
from fusion_msgs.msg import DetectionArray  
import message_filters

# Decodifica una imagen comprimida del mensaje ROS a imagen OpenCV
def decode_compressed_img(msg):
    np_arr = np.frombuffer(msg.data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        rospy.logwarn("decode_compressed_img: imagen es None")
    return img

# Dibuja las bounding boxes de un DetectionArray sobre la imagen dada
def draw_bboxes(img, detection_array, color=(0,255,0)):
    if img is None:
        return
    for detection in detection_array.detections:
        bbox = detection.bbox
        label = f"{detection.class_id} ({detection.confidence:.2f})"
        # Dibuja el cuadro de la bounding box
        cv2.rectangle(img, (int(bbox.xmin), int(bbox.ymin)),
                      (int(bbox.xmax), int(bbox.ymax)), color, 2)
        # Escribe la etiqueta de la clase encima de la bounding box
        cv2.putText(img, label, (int(bbox.xmin), max(int(bbox.ymin) - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

# Callback sincronizado para visualizar detecciones RGB y térmicas
def callback(det_thermal_array, det_webcam_array):
    # Extraer imágenes embebidas en los mensajes DetectionArray
    img_thermal = decode_compressed_img(det_thermal_array.image)
    img_webcam = decode_compressed_img(det_webcam_array.image)
    if img_thermal is None or img_webcam is None:
        rospy.logwarn("Una de las imágenes es None, salto esta iteración")
        return

    # Redimensionar ambas imágenes a la misma altura
    height = min(img_thermal.shape[0], img_webcam.shape[0])
    img_thermal_resized = cv2.resize(img_thermal,
                                     (int(img_thermal.shape[1] * height / img_thermal.shape[0]), height))
    img_webcam_resized = cv2.resize(img_webcam,
                                    (int(img_webcam.shape[1] * height / img_webcam.shape[0]), height))

    # Combinar ambas imágenes horizontalmente
    combined_img = np.hstack((img_thermal_resized, img_webcam_resized))

    # offset en el eje X para dibujar bounding boxes de la webcam en la imagen combinada (queda a la derecha)
    offset_x = img_thermal_resized.shape[1]

    # Dibujar bounding boxes térmicas (en rojo) en el lado izquierdo
    draw_bboxes(combined_img[:, :offset_x], det_thermal_array, color=(0, 0, 255))

    # Dibujar bounding boxes RGB (verde) en el lado derecho con desplazamiento en X
    for detection in det_webcam_array.detections:
        bbox = detection.bbox
        xmin = int(bbox.xmin) + offset_x    # para imprimir en el lado derecho
        xmax = int(bbox.xmax) + offset_x    # para imprimir en el lado derecho
        ymin = int(bbox.ymin)
        ymax = int(bbox.ymax)
        label = f"{detection.class_id} ({detection.confidence:.2f})"
        cv2.rectangle(combined_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(combined_img, label, (xmin, max(ymin - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Mostrar el resultado en una ventana pop-up de OpenCV
    cv2.imshow("Fusión Térmica y RGB con Detecciones", combined_img)
    cv2.waitKey(1)

def main():
    rospy.init_node("fusion_yolo_compressed", anonymous=True)

    # Suscriptores a detecciones RGB y térmicas
    sub_det_thermal = message_filters.Subscriber("/thermal_detections", DetectionArray)
    sub_det_webcam = message_filters.Subscriber("/rgb_detections", DetectionArray)

    # Sincronización de detecciones con ATS
    ats = message_filters.ApproximateTimeSynchronizer(
        [sub_det_thermal, sub_det_webcam],
        queue_size=30,
        slop=100.0  # margen temporal muy amplio para asegurar coincidencias
    )
    ats.registerCallback(callback)

    rospy.loginfo("Fusionando imágenes comprimidas y detecciones YOLO...")
    rospy.spin()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
