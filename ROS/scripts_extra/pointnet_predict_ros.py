#!/usr/bin/env python3
"""
===========================================================================
Nombre del archivo: pointnet_predict_ros.py

Descripción:
    Adaptación del script para comprobar la clase más probable de ModelNet40 de un
    clúster obtenido tras voxelización y DBSCAN al entorno de ROS. 

Funcionalidades:
    1. Escucha nubes de puntos segmentadas desde ROS (`/segmented_cloud`).
    2. Preprocesa las nubes para adaptarlas al formato requerido por PointNet.
    3. Ejecuta la clasificación y muestra el Top-5 de clases más probables.
    4. Muestra visualmente la nube en una ventana con Open3D.

Requisitos:
    - ROS (rospy, sensor_msgs)
    - Python 3.8+
    - NumPy
    - PyTorch
    - Open3D
    - learning3d (modelo PointNet y Classifier)

Uso:
    - Asegúrate de tener el modelo Pointnet preentrenado en la ruta esperada.
    - Lanza este nodo mientras se publican nubes en `/segmented_cloud`.

Autor: Ismael González Durán  
Fecha: 2025
===========================================================================
"""

import rospy
import sensor_msgs.msg
from sensor_msgs import point_cloud2
import open3d as o3d
import numpy as np
import torch
from learning3d.models import PointNet, Classifier
from std_msgs.msg import String
import copy

# Lista de clases del dataset ModelNet40
modelnet_classes = [
    "xbox", "wardrobe", "vase", "tv_stand", "toilet", "tent", "table", "stool", 
    "stairs", "sofa", "sink", "range_hood", "radio", "plant", "piano", "person", 
    "night_stand", "monitor", "mantel", "laptop", "lamp", "keyboard", "guitar", 
    "glass_box", "flower_pot", "dresser", "door", "desk", "curtain", "cup", 
    "cone", "chair", "car", "bowl", "bottle", "bookshelf", "bench", "bed", 
    "bathtub", "airplane"
]

# --- Cargar modelo PointNet preentrenado (cambiar ruta) del repositorio learning3d ---
model = Classifier(feature_model=PointNet(emb_dims=1024, use_bn=True))
model.load_state_dict(torch.load("./ruta_pointnet.t7", map_location="cpu"))   # Carga los pesos
model.eval()    # Modo evaluación (desactiva dropout y el modo entrenamiento)

# Procesa una nube de puntos y la clasifica
def process_point_cloud(msg):
    # Convertir el mensaje PointCloud2 en un array de puntos (x, y, z)
    pc_data = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
    points = np.array(list(pc_data), dtype=np.float32)

    # Normalización (centrado y escala)
   points -= np.mean(points, axis=0)   # Centrar nubes de puntos en el origen
    points /= np.max(np.linalg.norm(points, axis=1))    # Escalar a la unidad

    # Asegurar que haya al menos 1024 puntos, se rellena duplicando puntos si no hay suficientes
    if len(points) < 1024:
        indices = np.random.choice(len(points), 1024 - len(points), replace=True)
        points = np.concatenate([points, points[indices]])

    # --- Inferencia: predecir clase ---
    with torch.no_grad(): # No calcular gradientes (no se modifica el modelo)
        logits = model(torch.tensor(points).unsqueeze(0)) # Añadir dimensión batch para procesar varios archivos
        probs = torch.softmax(logits, dim=1) # Obtener probabilidades
        top5_probs, top5_indices = torch.topk(probs, 5) # Nos quedamos con las cinco probabilidades más altas

    # --- Mostrar resultados ---
    rospy.loginfo("\nTop 5 predicciones más probables:")
    for i in range(5):
        class_idx = top5_indices[0][i].item()   # índice de la clase en el dataset ModelNet40
        class_name = modelnet_classes[class_idx]    # nombre de la clase en el dataset ModelNet40
        confidence = top5_probs[0][i].item() * 100  # Porcentaje de confianza
        print(f"{i+1}. {class_name}: {confidence:.2f}%")

    # Mostrar la predicción principal
    pred_class = top5_indices[0][0].item()
    rospy.loginfo(f"\nPredicción principal: {modelnet_classes[pred_class]} ({top5_probs[0][0].item():.1%} confianza)")

# Visualiza una nube de puntos en una ventana 3D usando Open3D
def show_clusters(msg):
    pc_data = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
    points = np.array(list(pc_data), dtype=np.float32)

    # Crear objeto PointCloud de Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Mostrarlo en ventana interactiva de Open3D
    o3d.visualization.draw_geometries(
        [pcd],
        window_name="Clusters",
        width=800,
        height=600,
        left=50,
        top=50
    )

# Inicializa el nodo y suscriptores
def listener():
    rospy.init_node('point_cloud_listener', anonymous=True)

    # Suscriptores al mismo tópico pero diferentes callbacks (uno para visualizar, otro para clasificar)
    rospy.Subscriber('/segmented_cloud', sensor_msgs.msg.PointCloud2, process_point_cloud)
    rospy.Subscriber('/segmented_cloud', sensor_msgs.msg.PointCloud2, show_clusters)

    rospy.spin()

if __name__ == '__main__':
    listener()
