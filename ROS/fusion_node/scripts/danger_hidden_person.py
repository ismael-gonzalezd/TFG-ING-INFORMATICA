#!/usr/bin/env python3
"""
===========================================================================
Nombre del archivo: danger_hidden_person.py

Descripción:
    Nodo de ROS que realiza la fusión de detecciones RGB y térmicas. Está enfocado a
    alertar cuando haya personas que aparezcan en térmica, pero no en RGB para indicar
    un posible ocultamiento.

Funcionalidades:
    1. Fusiona detecciones RGB y térmicas usando homografía y reglas por clase.
    2. Detecta personas no visibles en RGB pero sí en térmica (alerta de ocultamiento).
    3. Dibuja resultados en imágenes, publica detecciones y lanza alertas visuales y sonoras.

Autor: Ismael González Durán  
Fecha: 2025
===========================================================================
"""
import rospy
from sensor_msgs.msg import CompressedImage
from message_filters import ApproximateTimeSynchronizer, Subscriber as WaitSub
import cv2
import os
import numpy as np
from fusion_msgs.msg import DetectionArray, Detection, BoundingBox as BBox
from utils_vision import encode_compressed_img, decode_compressed_img, transformar_bbox, calcular_iou, save_csv, image_sync_callback

# --- Variables globales y configuración ---
# Contador de fotogramas procesados 
frame_counter = 0

# Clases que deben controlarse para buscar ocultamiento
CLASES_PRIORIDAD_TERMICA = ["person_standing", "person_lying"]

# Matriz de homografía (precalculada entre térmica y RGB)
H = np.array([
    [7.64215087e-01, -2.69001316e-02, 6.03922905e+01],
    [5.29528562e-03,  7.37333274e-01, -1.07828857e+01],
    [3.25468646e-05, -6.22497906e-05,  1.00000000e+00]
], dtype=np.float64)
H_inv = np.linalg.inv(H)

# Muestra dos imágenes (RGB y térmica) lado a lado durante 5 segundos en una ventana emergente.
def mostrar_lado_a_lado(img1, img2, title="Alert"):
    # Altura y anchura de foto combinada
    altura_max = max(img1.shape[0], img2.shape[0])
    ancho_total = img1.shape[1] + img2.shape[1]
    combinada = np.zeros((altura_max, ancho_total, 3), dtype=np.uint8)

    # La imagen combinada se muestra durante cinco segundos con el mensaje de alerta y a los cinco segundos 
    # se cierra automáticamente  
    combinada[:img1.shape[0], :img1.shape[1], :] = img1
    combinada[:img2.shape[0], img1.shape[1]:, :] = img2
    cv2.imshow(title, combinada)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()

# Callback principal: procesa detecciones y alerta si hay personas ocultas
def detections_callback(det_thermal_array, det_webcam_array):
    global frame_counter, pub_fusion_detections, folder_rgb, folder_thermal

    # Decodifica las imágenes RGB y térmica a OpenCV
    img_rgb = decode_compressed_img(det_webcam_array.image)
    img_th = decode_compressed_img(det_thermal_array.image)
    if img_rgb is None or img_th is None:
        rospy.logwarn("Could not decode image")
        return

    # Inicializa estructura de detecciones fusionadas
    fusionadas = DetectionArray()
    fusionadas.header = det_webcam_array.header
    usadas_thermal = set() # para evitar usar la misma detección térmica varias veces

    # Procesamiento de detecciones RGB
    for det_rgb in det_webcam_array.detections:
        # Guardado de los datos de la detección (bounding box como tupla, índice de clase y confianza del modelo)
        bbox_rgb = (int(det_rgb.bbox.xmin), int(det_rgb.bbox.ymin), int(det_rgb.bbox.xmax), int(det_rgb.bbox.ymax))
        class_id = det_rgb.class_id
        conf_rgb = det_rgb.confidence

        # Dibuja la detección RGB sobre la imagen en color verde
        cv2.rectangle(img_rgb, (bbox_rgb[0], bbox_rgb[1]), (bbox_rgb[2], bbox_rgb[3]), (0, 255, 0), 2)
        cv2.putText(img_rgb, f"{class_id} ({conf_rgb:.2f})", (bbox_rgb[0], max(bbox_rgb[1] - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Agrega la detección RGB directamente al array fusionado
        det = Detection(class_id=class_id, confidence=conf_rgb,
                        bbox=BBox(xmin=bbox_rgb[0], ymin=bbox_rgb[1], xmax=bbox_rgb[2], ymax=bbox_rgb[3]))
        fusionadas.detections.append(det)

    # Buscar posibles personas ocultas solo detectadas en térmica
    for j, det_t in enumerate(det_thermal_array.detections):
        # ignorar detecciones que no sean personas
        if det_t.class_id not in CLASES_PRIORIDAD_TERMICA:
            continue 

        # Proyectar bounding box térmica al plano RGB 
        bbox_t_proj = transformar_bbox(det_t.bbox, H)

        # Ver si hay una detección RGB con la misma clase y suficiente IoU
        match_found = False
        for det_rgb in det_webcam_array.detections:
            # ignorar las de clases diferentes
            if det_rgb.class_id != det_t.class_id:
                continue
            
            # Obtener bounding box térmica y calcular IoU. Si es mayor que 0.1 se considera que la persona no está oculta
            bbox_rgb = (int(det_rgb.bbox.xmin), int(det_rgb.bbox.ymin), int(det_rgb.bbox.xmax), int(det_rgb.bbox.ymax))
            if calcular_iou(bbox_rgb, bbox_t_proj) > 0.1:
                match_found = True
                break

        # Si no se encontró correspondencia, se lanza alerta
        if not match_found:
            # Obtener boundingbox térmica original 
            x1t, y1t, x2t, y2t = int(det_t.bbox.xmin), int(det_t.bbox.ymin), int(det_t.bbox.xmax), int(det_t.bbox.ymax)
            
            # Dibujar en imagen térmica y lanzar alerta (en la imagen, en la termical y vía audio usando el comando espeak)
            cv2.rectangle(img_th, (x1t, y1t), (x2t, y2t), (0, 0, 255), 2)
            cv2.putText(img_th, "ALERTA: Persona oculta", (x1t, max(y1t - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            rospy.logwarn("\u00a1Persona detectada en térmica pero no en RGB!")
            mostrar_lado_a_lado(img_rgb, img_th, title="Persona posiblemente escondida")
            os.system('espeak -s 150 "Warning. Hidden person detected."')

    # Publicar detecciones y guardar imágenes resultantes
    pub_fusion_detections.publish(fusionadas)
    cv2.imwrite(os.path.join(folder_rgb, f"fusion_rgb_{frame_counter}.png"), img_rgb)
    cv2.imwrite(os.path.join(folder_thermal, f"fusion_thermal_{frame_counter}.png"), img_th)
    frame_counter += 1

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
