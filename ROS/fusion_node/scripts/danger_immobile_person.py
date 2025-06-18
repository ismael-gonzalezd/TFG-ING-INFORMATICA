#!/usr/bin/env python3
"""
===========================================================================
Nombre del archivo: danger_immobile_person.py

Descripción:
    Nodo ROS para la fusión de detecciones entre cámara RGB y térmica, que lanza alertas
    en presencia de personas tumbadas e inmóviles. Aplica una homografía
    para proyectar los bounding boxes térmicos al plano de la cámara RGB.

Funcionalidades:
    - Fusiona detecciones térmicas y RGB priorizando ciertas clases.
    - Realiza seguimiento de personas tumbadas para alertar si están inmóviles.
    - Dibuja cajas y etiquetas en las imágenes procesadas.
    - Guarda imágenes y publica alertas visuales y auditivas.

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
import csv
from fusion_msgs.msg import DetectionArray, Detection, BoundingBox as BBox
from utils_vision import encode_compressed_img, decode_compressed_img, transformar_bbox, calcular_iou, save_csv, image_sync_callback

# Contador de frames procesados
frame_counter = 0

# Clases con prioridad en detección térmica y RGB
CLASES_PRIORIDAD_TERMICA = ["person_standing", "person_lying"]
CLASES_PRIORIDAD_RGB = ["chair", "laptop", "bin", "backpack"]

# PARÁMETROS PARA EL SEGUIMIENTO DE PERSONAS TUMBADAS
IOU_MATCH_THRESHOLD = 0.5   # Solapamiento mínimo para vincular una persona tumbada a un tracker existente
MIN_FRAMES_LYING = 10   # Mínimo de fotogramas que debe estar una persona tumbada para ser considerada como inmóvil
MAX_POSITION_SHIFT = 20 # Movimiento máximo permitido para considerar una persona como inmóvil 
RENOTIFY_TIMEOUT = 10   # Intervalo de segundos entre alertas por personas tumbadas inmóviles 
lying_trackers = {} # Diccionario de trackers (tiempo de detección -> info. detección)

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

def desplazamiento(box1, box2):
    return sum([abs(a - b) for a, b in zip(box1, box2)])

# ================================================================
# Callback de fusión de detecciones con alerta de personas inmóviles
# ================================================================
def detections_callback(det_thermal_array, det_webcam_array):
    global frame_counter, pub_fusion_detections, folder_rgb, folder_thermal, lying_trackers

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

        # Variables para encontrar la detección térmica más cercana
        best_iou, best_det_t, best_idx = 0, None, -1

        # Recorremos las detecciones térmicas buscando coincidencia con la detección RGB
        for j, det_t in enumerate(det_thermal_array.detections):
            # Solo consideramos detecciones de la misma clase que no se hayan usado antes
            if det_t.class_id != class_id or j in usadas_thermal:
                continue
            
            # Transformamos la bounding box térmica al plano RGB usando homografía y calculamos su IoU
            bbox_t_proj = transformar_bbox(det_t.bbox, H)
            iou = calcular_iou(bbox_rgb, bbox_t_proj)
            
            # Guardamos la mejor coincidencia térmica si el IoU es mayor que el caso anterior
            if iou > best_iou:
                best_iou, best_det_t, best_idx = iou, det_t, j

        # Solo consideramos el emparejamiento si el solapamiento (IoU) es significativo (> 0.3)
        if best_iou > 0.3 and best_det_t:
            usadas_thermal.add(best_idx)    # Marcamos la detección térmica como utilizada
            conf_t = det_rgb.confidence, best_det_t.confidence  # Obtenemos la confianza del modelo térmico

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
            det = Detection(class_id=class_id, confidence=conf, bbox=BBox(xmin=x1, ymin=y1, xmax=x2, ymax=y2))
            fusionadas.detections.append(det)

            # SEGUIMIENTO DE PERSONAS TUMBADAS
             if class_id == "person_lying":
                matched_id = None
                for track_id, data in lying_trackers.items():
                    if calcular_iou((x1, y1, x2, y2), data["bbox"]) > IOU_MATCH_THRESHOLD:
                        matched_id = track_id
                        break
                
                # Si existe ya un tracker para esta detección, actualizamos su información 
                if matched_id:
                    lying_trackers[matched_id]["bbox"] = (x1, y1, x2, y2)   # Actualizamos la caja intersección térmica-RGB
                    lying_trackers[matched_id]["frames"] += 1               # Aumentamos el número de fotogramas que lleva activo el seguimiento
                    lying_trackers[matched_id]["bbox_rgb"] = bbox_rgb       # Actualizamos la bounding box RGB
                    lying_trackers[matched_id]["bbox_thermal"] = (int(best_det_t.bbox.xmin), int(best_det_t.bbox.ymin), int(best_det_t.bbox.xmax), int(best_det_t.bbox.ymax)) # Actualizamos por la bounding box térmica

                # Si no existe un tracker, lo creamos de cero
                else:
                    # Usamos de clave, los nanosegundos que han pasado
                    track_id = str(rospy.Time.now().to_nsec())
                    # Añadimos la información de la detección 
                    lying_trackers[track_id] = {
                        "bbox": (x1, y1, x2, y2),                   # Bounding Box intersección térmica-RGB
                        "initial_position": (x1, y1, x2, y2),       # Posición inicial (al empezar el tracking), no se actualiza en detecciones posteriores
                        "frames": 1,                                # Indicamos que el seguimiento solo lleva 1 fotograma de momento
                        "notified": False,                          # Indicamos que no ha saltado ninguna alerta aún sobre esta detección
                        "last_updated": rospy.Time.now().to_sec(),  # Indicamos que la última actualización fue ahora (solo se cambia para nuevas alertas)
                        "bbox_rgb": bbox_rgb,                       # Añadimos la bounding box RGB original
                        "bbox_thermal": (int(best_det_t.bbox.xmin), int(best_det_t.bbox.ymin), int(best_det_t.bbox.xmax), int(best_det_t.bbox.ymax))        # Añadimos la bounding box térmica original (sin proyectar)
                    }
            
            # Dibujamos bounding boxes en RGB 
            x1r, y1r, x2r, y2r = bbox_rgb
            cv2.rectangle(img_rgb, (x1r, y1r), (x2r, y2r), (255, 0, 255), 2)
            cv2.putText(img_rgb, f"{class_id} ({conf:.2f})", (x1r, max(y1r - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

            # Dibujamos bounding boxes en térmica
            x1t, y1t, x2t, y2t = int(best_det_t.bbox.xmin), int(best_det_t.bbox.ymin), int(best_det_t.bbox.xmax), int(best_det_t.bbox.ymax)
            cv2.rectangle(img_th, (x1t, y1t), (x2t, y2t), (255, 0, 255), 2)
            cv2.putText(img_th, f"{class_id} ({conf:.2f})", (x1t, max(y1t - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    # EVALUACIÓN DE LA INMOVILIDAD (revisión de personas tumbadas con seguimiento)
    for track_id, data in lying_trackers.items():

        # Se activa la alerta si la persona lleva más de MIN_FRAMES_LYING tumbada y si el desplazamiento acumulado de su bounding box no supera MAX_POSITION_SHIFT
        if data["frames"] >= MIN_FRAMES_LYING and desplazamiento(data["bbox"], data["initial_position"]) < MAX_POSITION_SHIFT:
            
            # Se añade una alerta visual con un rectángulo naranja para marcar que se esta haciendo el tracking
            x1, y1, x2, y2 = map(int, data["bbox_rgb"])
            x1t, y1t, x2t, y2t = map(int, data["bbox_thermal"])
            cv2.rectangle(img_rgb, (x1 - 5, y1 -7), (x2 +5, y2 +5), (0,165,255), 2)
            cv2.rectangle(img_th, (x1t, y1t), (x2t, y2t), (0,165,255), 2)
            now = rospy.Time.now().to_sec()

            # Solo se lanza audio si no se ha hecho anteriormente o ha pasado suficiente tiempo desde el ultimo aviso. También se añade el texto de alerta en naranja a las imagenes
            if not data["notified"] or (now - data["last_updated"] > RENOTIFY_TIMEOUT):
                # Escribir mensjae en imagen térmica y lanzar alerta (en la imagen, en la termical y vía audio usando el comando espeak)
                cv2.putText(img_rgb, "DANGER: Immobile person detected!", (x1r, y1r - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2)
                cv2.putText(img_th, "DANGER: Immobile person detected!", (x1t, y1t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,165,255), 2)
                mostrar_lado_a_lado(img_rgb, img_th)
                
                os.system('espeak -s 150 "DANGER: Immobile person detected!"')
                rospy.logwarn(f"Immobile person detected at bbox: {data['bbox']}")
                # Actualizamos la información del tracker para evitar alertas repetidas
                data["notified"] = True        # Especificamos que ya se lanzó la alerta al menos una vez
                data["last_updated"] = now     # Actualizamos la hora de la última alerta

    
    # Limpiamos los trackers para evitar almacenar información muy antigua (la persona puede haber sido atendida o haberse levantado)
    lying_trackers = {k: v for k, v in lying_trackers.items() if v["frames"] < 100}

    # Publicamos los resultados en un tópico y guardamos las imágenes en el disco
    pub_fusion_detections.publish(fusionadas)
    cv2.imwrite(os.path.join(folder_rgb, f"fusion_rgb_{frame_counter}.png"), img_rgb)
    cv2.imwrite(os.path.join(folder_thermal, f"fusion_thermal_{frame_counter}.png"), img_th)
    frame_counter += 1

def mostrar_lado_a_lado(img1, img2, title="Alert"):
    # Altura y anchura de la imagne combinada
    altura_max = max(img1.shape[0], img2.shape[0])
    ancho_total = img1.shape[1] + img2.shape[1]

    # Crear imagen de fondo
    combinada = np.zeros((altura_max, ancho_total, 3), dtype=np.uint8)
    combinada[:img1.shape[0], :img1.shape[1], :] = img1
    combinada[:img2.shape[0], img1.shape[1]:, :] = img2

    # La imagen se muestra por pantalla durante 5 segundos, luego se cierra automaticamente
    cv2.imshow(title, combinada)
    cv2.imwrite(os.path.join(folder_alert, f"alerta_{frame_counter}.png"), combinada)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()


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

