"""
===========================================================================
Nombre del archivo: pointnet_predict.py

Descripción:
    Este script clasifica una nube de puntos 3D (archivo .ply) utilizando un
    modelo PointNet preentrenado. Carga el modelo, normaliza la nube y predice la
    clase del objeto en la nube, mostrando las 5 clases más probables con su 
    porcentaje de confianza.

Funcionalidades:
    - Lectura de nube de puntos en formato .ply.
    - Normalización de coordenadas
    - Carga de un modelo PointNet preentrenado.
    - Predicción de la clase del objeto con top-5 resultados.
    - Soporte para 40 clases de ModelNet40.

Requisitos:
    - Python 3.8
    - Open3D
    - NumPy
    - PyTorch
    - learning3d (https://github.com/qq456cvb/learning3d)

Uso:
    python3 pointnet_predict.py ruta_archivo.ply

Autor: Ismael González Durán
Fecha: 2025
===========================================================================
"""

import open3d as o3d
import numpy as np
import torch
import sys
from learning3d.models import PointNet, Classifier

if len(sys.argv) != 2:
    print("Uso: python3 pointnet_predict.py ruta_archivo.ply")
    sys.exit(1)

# Cargar nube a partir de la ruta del archivo .ply
ply_path = sys.argv[1]
cloud = o3d.io.read_point_cloud(ply_path)

# Lista de clases del dataset ModelNet40
modelnet_classes = [
    "xbox", "wardrobe", "vase", "tv_stand", "toilet", "tent", "table", "stool", 
    "stairs", "sofa", "sink", "range_hood", "radio", "plant", "piano", "person", 
    "night_stand", "monitor", "mantel", "laptop", "lamp", "keyboard", "guitar", 
    "glass_box", "flower_pot", "dresser", "door", "desk", "curtain", "cup", 
    "cone", "chair", "car", "bowl", "bottle", "bookshelf", "bench", "bed", 
    "bathtub", "airplane"
]

# --- Normalización de la nube de puntos ---
points = np.asarray(cloud.points, dtype=np.float32) # Convertir a array de puntos NumPy
points -= np.mean(points, axis=0)   # Centrar nubes de puntos en el origen
points /= np.max(np.linalg.norm(points, axis=1))    # Escalar a la unidad

# Asegurar que haya al menos 1024 puntos, se rellena duplicando puntos si no hay suficientes
if len(points) < 1024:
    indices = np.random.choice(len(points), 1024 - len(points), replace=True)
    points = np.concatenate([points, points[indices]])

# --- Cargar modelo PointNet preentrenado (cambiar ruta) del repositorio learning3d ---
model = Classifier(feature_model=PointNet(emb_dims=1024, use_bn=True))
model.load_state_dict(torch.load("./ruta_pointnet.t7", map_location="cpu"))   # Carga los pesos
model.eval()    # Modo evaluación (desactiva dropout y el modo entrenamiento)

# --- Inferencia: predecir clase ---
# INFERENCIA: se calcula la probabilidad de cada clase
with torch.no_grad():
    logits = model(torch.tensor(points).unsqueeze(0))  # Se añade dimensión de batch para procesar varios archivos
    probs = torch.softmax(logits, dim=1)               # Se convierte a probabilidad
    top5_probs, top5_indices = torch.topk(probs, 5)    # Se extraen las 5 clases más probables

# --- Mostrar resultados ---
print("\nTop 5 predicciones más probables:")
for i in range(5):
    class_idx = top5_indices[0][i].item()   # índice de la clase en el dataset ModelNet40
    class_name = modelnet_classes[class_idx]    # nombre de la clase en el dataset ModelNet40
    confidence = top5_probs[0][i].item() * 100  # Porcentaje de confianza
    print(f"{i+1}. {class_name}: {confidence:.2f}%")

pred_class = top5_indices[0][0].item()
print(f"\nPredicción principal: {modelnet_classes[pred_class]} ({top5_probs[0][0].item():.1%} confianza)")

