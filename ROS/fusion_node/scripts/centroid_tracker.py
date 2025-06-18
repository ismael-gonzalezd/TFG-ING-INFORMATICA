"""
===============================================================================
Nombre del archivo: centroid_tracker.py

Descripción:
    Implementación de un rastreador simple basado en centroides para objetos detectados,
    específicamente para seguir vehículos (clase "car") en fotogramas diferentes.
    Se asigna un ID único a cada nuevo objeto detectado y se reasigna si su posición
    se mantiene dentro de un umbral de distancia máxima.

Uso:
    Puede utilizarse junto con un sistema de detección que proporcione bounding boxes
    con atributos bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax y class_id.

Autor: Ismael González Durñan
Fecha: 2025
===============================================================================
"""

import numpy as np 

# Función auxiliar para calcular el centro de una bounding box
def calcular_centro(bbox):
    cx = (bbox.xmin + bbox.xmax) / 2
    cy = (bbox.ymin + bbox.ymax) / 2
    return np.array([cx, cy])

# ===============================
# Clase para rastrear objetos basándose en sus centroides
class CentroidTracker:
    def __init__(self, max_distance=50, tracked_class = "car"):
        self.next_id = 0                        # id que se asignará al siguiente objeto nuevo
        self.objects = {}                       # diccionario con el ID del objeto y su centro actual
        self.max_distance = max_distance        # distancia máxima para asociar una detección a un ID ya existente
        self.tracked_class = tracked_class    # clase de objeto que se está siguiendo

    def update(self, detecciones):
        """
        Actualiza el rastreador con nuevas detecciones.
        Asocia detecciones nuevas con objetos existentes o asigna nuevos IDs.
        """
        nuevos_objetos = {}         # Nuevos objetos tras la actualización
        ids_asignados = {}          # Mapeo índice detección -> ID
        centroides_actuales = []    # Lista de (índice, centroide)

        # Recorremos las detecciones y solo consideramos las de clase 'car'
        for idx, det in enumerate(detecciones):
            # Si corresponde a un objeto de la clase buscada (car), calculamos su centroide 
            if det.class_id == self.tracked_class:
                c = calcular_centro(det.bbox)
                centroides_actuales.append((idx, c))
        
        # Si no hay objetos previos, asignamos nuevos IDs directamente
        if not self.objects:
            for idx, c in centroides_actuales:
                nuevos_objetos[self.next_id] = c        # guardamos el centroide de cada objeto trackeado
                ids_asignados[idx] = self.next_id       # generamos un id para la detección
                self.next_id += 1                       
       
        # Si hay objetos previos, tenemos que comprobar si la detección corresponde a un objeto ya trackeado
        else:
            ids_disponibles = list(self.objects.keys())     # todos los ids disponibles
            usados = set()      # ids ya asignados en esta llamada a "update"

            # Para cada centroide nuevo, buscamos la mejor coincidencia con los trackers existentes
            for idx, c in centroides_actuales:
                min_dist = float("inf")
                best_id = None

                # Comparamos con cada objeto registrado
                for obj_id in ids_disponibles:
                    # usamos la distancia euclidea 
                    d = np.linalg.norm(c - self.objects[obj_id])

                    # Si la caja está más cerca que la anterior, la guardamos como mejor candidata a "emparejar"
                    if d < min_dist and d < self.max_distance:
                        min_dist = d
                        best_id = obj_id

                # Se ha encontrado un objeto suficientemente cercano, podemos reutilizar el id y actualizar su centroide        
                if best_id is not None:
                    nuevos_objetos[best_id] = c
                    ids_asignados[idx] = best_id
                    usados.add(best_id)
                
                # Si no se ha encontrado un objeto suficientemente cercano, lo consideramos una nueva detección (generamos un id nuevo)
                else:
                    nuevos_objetos[self.next_id] = c
                    ids_asignados[idx] = self.next_id
                    self.next_id += 1

        # Actualizamos el estado interno del rastreador con los nuevos objetos detectados
        self.objects = nuevos_objetos

        # Se devuelve el mapeo de detecciones a ids y el estado actualizado del rastreador
        return ids_asignados, nuevos_objetos