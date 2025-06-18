#!/usr/bin/env python3
"""
===================================================================================
Nombre del archivo: danger_car.py

Descripcion:
    Nodo ROS que detecta situaciones de peligro cuando un coche se mueve en presencia
    de personas. Fusiona detecciones RGB y térmicas, rastrea vehículos por centroides,
    y guarda evidencia visual cuando se dispara una alerta.

Funcionalidades:
    - Detección de coches/personas en streams RGB y térmicos.
    - Rastreo de coches basado en centroides.
    - Verificación si una persona está fuera de un coche (en peligro).
    - Detección de movimiento y disparo de alertas sonoras.
    - Guardado de capturas RGB y térmicas de la escena.

Autor: Ismael González Durán
Fecha: 2025
===================================================================================
"""
import rospy
from sensor_msgs.msg import CompressedImage
from message_filters import ApproximateTimeSynchronizer
from message_filters import Subscriber as WaitSub
import cv2
import os
import numpy as np
from collections import deque
from fusion_msgs.msg import DetectionArray, Detection, BoundingBox as BBox
from centroid_tracker import CentroidTracker, calcular_centro
from utils_vision import image_sync_callback, decode_compressed_img

# Contador de frames procesados
frame_counter = 0

# PARÁMETROS PARA EL SEGUIMIENTO DE VEHICULOS
tracker = CentroidTracker()             # Instancia del rastreador por centroides
posiciones_previas = {}                 # Almacena la última posición de cada coche (id - centroide)
cola_previas = deque(maxlen=10)         # Cola circular con los últimos 10 frames previos a la alerta 
frames_post_event = 0                   # Contaodr para guardar frames después de la alerta
contador_peligro = 0                    # Contador de alertas producidas
peligro_dir_actual = ""                 # Ruta donde se guardan las imagenes de la ultima alerta
esperar_post_peligro = 0                # Número de fotogramas que quedan por guardar tras una alerta
MIN_TIME_BETWEEN_ALERTS = 8             # Tiempo mínimo (en segundos) entre dos alertas
ultimo_alerta_ts = 0                    # Timestamp de la ultima alerta producida

def bbox_pasajero(bbox_coche, bbox_persona):
    """
    Verifica si una persona está contenida dentro del bounding box de un coche
    Útil para evitar falsas alarmas si la persona está dentro del vehículo (se entiende
    que es pasajera/conductora del coche)
    """
    return (bbox_persona.xmin >= bbox_coche.xmin and
            bbox_persona.ymin >= bbox_coche.ymin and
            bbox_persona.xmax <= bbox_coche.xmax and
            bbox_persona.ymax <= bbox_coche.ymax)

# ======================================
# CALLBACK PRINCIPAL DE DETECCIÓN Y ALERTA
def detections_callback(det_thermal_array, det_webcam_array):
    global frame_counter, pub_fusion_detections, folder_rgb
    global posiciones_previas, tracker
    global cola_previas, frames_post_event, contador_peligro, peligro_dir_actual
    global esperar_post_peligro, ultimo_alerta_ts

    # Decodifica las imágenes RGB y térmica a OpenCV
    img_rgb = decode_compressed_img(det_webcam_array.image)
    img_th = decode_compressed_img(det_thermal_array.image)
    if img_rgb is None or img_th is None:
        rospy.logwarn("No se pudo decodificar alguna imagen")
        return

    # Filtra detecciones por clase (coche y persona) para cada cámara
    coches_rgb = [det for det in det_webcam_array.detections if det.class_id == "car"]
    personas_rgb = [det for det in det_webcam_array.detections if det.class_id == "person"]
    coches_th = [det for det in det_thermal_array.detections if det.class_id == "car"]
    personas_th = [det for det in det_thermal_array.detections if det.class_id == "person"]

    # Actualiza el tracker con las detecciones de coches RGB
    ids_asignados, _ = tracker.update(coches_rgb)
    movimiento_detectado = False
    coches_en_movimiento = []

    # Detectar movimiento significativo de los coches
    for idx, det in enumerate(coches_rgb):
        # Obtenemos el identificador de cada detección del coche 
        if idx not in ids_asignados:
            continue
        coche_id = ids_asignados[idx]

        # calculamos el centroide de la detección actual
        centro_actual = calcular_centro(det.bbox)

        # Si el coche ya había sido rastreado, calculamos lo que se ha desplazado
        if coche_id not in posiciones_previas:
            posiciones_previas[coche_id] = centro_actual
            continue
        centro_prev = posiciones_previas[coche_id]
        desplazamiento = np.linalg.norm(centro_actual - centro_prev)
        posiciones_previas[coche_id] = centro_actual

        # Si se ha desplazado mas de 25px, marcamos el movimiento como detectado
        if desplazamiento > 25:
            movimiento_detectado = True
            coches_en_movimiento.append(det)

    # Filtrar personas térmicas que estén dentro de coches
    personas_th_filtradas = []
    for persona in personas_th:
        dentro_de_coche = False
        for coche in coches_th:
            if bbox_pasajero(coche.bbox, persona.bbox):
                dentro_de_coche = True
                break
        if not dentro_de_coche:
            personas_th_filtradas.append(persona)
    
    # Verificamos si hay peligro (movimiento detectado y peatones fuera de coches)
    hay_persona = len(personas_th_filtradas) > 0
    alerta_activada = movimiento_detectado and hay_persona

    ahora = rospy.Time.now().to_sec()
    tiempo_desde_ultima_alerta = ahora - ultimo_alerta_ts

    imagenes_a_guardar = []
    
    # Se dispara la alerta si hay peligroy si han pasado suficientes fotagramas y tiempo desde la ultima alerta
    if alerta_activada and esperar_post_peligro == 0 and tiempo_desde_ultima_alerta >= MIN_TIME_BETWEEN_ALERTS:
        contador_peligro += 1
        # Se avisa por audio de la existencia de un coche en movimiento
        os.system('espeak "DANGER: Car moving next to pedestrians"')

        # Guardamos las pruebas de imagen del peligro (10 fotos antes y la de la alerta)
        peligro_dir_actual = os.path.join(folder_rgb, f"peligro{contador_peligro}")
        os.makedirs(peligro_dir_actual, exist_ok=True)

        imagenes_a_guardar = [(img_rgb.copy(), img_th.copy(), coches_rgb.copy(), personas_rgb.copy(), coches_th.copy(), personas_th_filtradas.copy(), False)
                              for (img_rgb, img_th, coches_rgb, personas_rgb, coches_th, personas_th) in cola_previas]
        imagenes_a_guardar.append((img_rgb.copy(), img_th.copy(), coches_rgb.copy(), personas_rgb.copy(), coches_th.copy(), personas_th_filtradas.copy(), True))
        cola_previas.clear()
        frames_post_event = 10
        esperar_post_peligro = 10
        ultimo_alerta_ts = ahora
    
    # también guardamos las 10 imágenes posteriores al peligro
    elif esperar_post_peligro > 0:
        imagenes_a_guardar = [(img_rgb.copy(), img_th.copy(), coches_rgb.copy(), personas_rgb.copy(), coches_th.copy(), personas_th_filtradas.copy(), False)]
        frames_post_event -= 1
        esperar_post_peligro -= 1

    for img_rgb_saved, img_th_saved, dets_coches_rgb, dets_personas_rgb, dets_coches_th, dets_personas_th, es_disparadora in imagenes_a_guardar:
        # Dibujo en RGB (detecciones RGB)
        for det in dets_coches_rgb:
            x1, y1, x2, y2 = int(det.bbox.xmin), int(det.bbox.ymin), int(det.bbox.xmax), int(det.bbox.ymax)
            cv2.rectangle(img_rgb_saved, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_rgb_saved, "COCHE", (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        for det in dets_personas_rgb:
            x1, y1, x2, y2 = int(det.bbox.xmin), int(det.bbox.ymin), int(det.bbox.xmax), int(det.bbox.ymax)
            cv2.rectangle(img_rgb_saved, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img_rgb_saved, "PERSONA", (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Dibujo en térmica (detecciones térmicas)
        for det in dets_coches_th:
            x1, y1, x2, y2 = int(det.bbox.xmin), int(det.bbox.ymin), int(det.bbox.xmax), int(det.bbox.ymax)
            cv2.rectangle(img_th_saved, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_th_saved, "COCHE", (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        for det in dets_personas_th:
            x1, y1, x2, y2 = int(det.bbox.xmin), int(det.bbox.ymin), int(det.bbox.xmax), int(det.bbox.ymax)
            cv2.rectangle(img_th_saved, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img_th_saved, "PERSONA", (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Añadimos mensaje de PELIGRO en la imagen térmica y RGB en la que saltó la alarma 
        if es_disparadora:
            cv2.putText(img_rgb_saved, "PELIGRO!", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(img_th_saved, "PELIGRO!", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Ajuste de tamaño si RGB y térmica difieren
        if img_rgb_saved.shape[0] != img_th_saved.shape[0]:
            altura_rgb = img_rgb_saved.shape[0]
            ancho_th = int(img_th_saved.shape[1] * (altura_rgb / img_th_saved.shape[0]))
            img_th_saved = cv2.resize(img_th_saved, (ancho_th, altura_rgb))

        # Guardado de las memorias en disco
        cv2.imwrite(os.path.join(peligro_dir_actual, f"webcam_{frame_counter}.png"), img_rgb_saved)
        cv2.imwrite(os.path.join(peligro_dir_actual, f"thermal_{frame_counter}.png"), img_th_saved)
        frame_counter += 1

    # Añade el frame actual a la cola si no estamos en modo post-alerta
    if esperar_post_peligro == 0:
        cola_previas.append((img_rgb.copy(), img_th.copy(), coches_rgb.copy(), personas_rgb.copy(), coches_th.copy(), personas_th_filtradas.copy()))

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
