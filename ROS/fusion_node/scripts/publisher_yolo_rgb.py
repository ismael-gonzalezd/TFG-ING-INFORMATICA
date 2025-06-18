#!/usr/bin/env python3
"""
===========================================================================
Nombre del archivo: publisher_yolo_rgb.py

Descripción:
    Nodo de ROS que utiliza un modelo YOLOv8 para detectar objetos en imágenes
    RGB comprimidas recibidas desde un tópico. Publica todas las detecciones
    (clase, confianza y bounding box) en un mensaje personalizado tipo DetectionArray.

Funcionalidades:
    1. Suscribe a imágenes térmicas comprimidas desde el tópico `/rgb_img_coincidence`.
    2. Decodifica la imagen usando OpenCV.
    3. Infiere para detectar clases con un modelo YOLOv8 entrenado previamente.
    4. Publica las detecciones en el tópico `/rgb_detections` como mensajes tipo DetectionArray.

Requisitos:
    - ROS (rospy, sensor_msgs, std_msgs)
    - OpenCV (cv2)
    - NumPy
    - YOLOv8 (ultralytics)
    - cv_bridge
    - Mensajes personalizados: fusion_msgs.msg

Uso:
    - Asegúrate de que el modelo YOLO esté ubicado en la ruta indicada y en formato .pt.
    - Lanza este nodo mientras se publican imágenes térmicas comprimidas.

Autor: Ismael González Durán  
Fecha: 2025
===========================================================================
"""

import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
from ultralytics import YOLO
import numpy as np
import cv2
import os
from fusion_msgs.msg import DetectionArray, Detection, BoundingBox
from std_msgs.msg import Header

class YOLOThermalPublisher:
    def __init__(self):
        # Inicializa el nodo ROS
        rospy.init_node("yolo_thermal_node", anonymous=True)
        rospy.Time.now()

        # Puente para convertir entre mensajes ROS e imágenes OpenCV
        self.bridge = CvBridge()

        # Carga el modelo YOLOv8 desde la ruta del sistema de archivos
        model_path = os.path.expanduser("~/modelos/modelo_rgb.pt")
        self.model = YOLO(model_path)

        # Publicador de detecciones en el tópico especificado
        self.pub = rospy.Publisher("/rgb_detections", DetectionArray, queue_size=10)

        # Suscriptor al tópico de imagen térmica comprimida
        rospy.Subscriber("/rgb_img_coincidence", CompressedImage, self.image_callback)

    def image_callback(self, msg):
        """
        Callback que procesa cada imagen recibida, realiza inferencia con YOLO
        y publica las detecciones en el tópico correspondiente.
        """
        try:
            # Convierte la imagen comprimida a una imagen OpenCV
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # Realiza la detección usando el modelo YOLO
            results = self.model(cv_image)

            # Construye el mensaje de detección para ROS
            detection_array = DetectionArray()
            detection_array.header = Header()
            detection_array.header.stamp = rospy.Time.now()

            # Procesa cada resultado de detección
            for result in results:
                if result.boxes is None:
                    continue

                for box in result.boxes:
                    det = Detection()
                    det.header.stamp = rospy.Time.now()

                    # Asigna la clase detectada
                    det.class_id = result.names[int(box.cls[0])] if hasattr(result, "names") else str(int(box.cls[0]))
                    det.confidence = float(box.conf)
                    det.source = "RGB"

                    # Extrae y asigna las coordenadas del bounding box
                    x1, y1, x2, y2 = box.xyxy[0]
                    bbox = BoundingBox()
                    bbox.xmin = float(x1)
                    bbox.ymin = float(y1)
                    bbox.xmax = float(x2)
                    bbox.ymax = float(y2)
                    det.bbox = bbox

                    # Añade la detección al array
                    detection_array.detections.append(det)

            # Adjunta la imagen original al mensaje
            detection_array.image = msg

            # Publica las detecciones
            self.pub.publish(detection_array)

        except Exception as e:
            rospy.logerr(f"Error procesando imagen: {e}")

if __name__ == '__main__':
    try:
        YOLOThermalPublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
