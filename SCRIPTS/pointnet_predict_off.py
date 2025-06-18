"""
===========================================================================
Nombre del archivo: pointnet_predict_off.py

Descripción:
    Este script clasifica una nube de puntos 3D, pero en formato de archivo .off
    utilizando un modelo PointNet preentrenado. Carga el modelo, normaliza la nube y 
    predice la clase del objeto en la nube, mostrando las 5 clases más probables con su 
    porcentaje de confianza.

Funcionalidades:
    - Lectura de nube de puntos en formato .off.
    - Normalización de coordenadas
    - Predicción de la clase del objeto con top-5 resultados usando un modelo PointNet preentrenado.
    - Soporte para 40 clases de ModelNet40.
    - Guarda imagen 3D de la nube de puntos como snapshot

Requisitos:
    - Open3D
    - NumPy
    - PyTorch
    - learning3d (https://github.com/qq456cvb/learning3d)

Uso:
    python3 pointnet_predict_off.py ruta_archivo.off

Autor: Ismael González Durán
Fecha: 2025
===========================================================================
"""

import numpy as np
import torch
import sys
import os
import matplotlib.pyplot as plt
from learning3d.models import PointNet, Classifier

# Función para cargar la nube de puntos desde un archivo .off
def load_off_points(path):
    with open(path, 'r') as f:
        lines = f.readlines()

    if lines[0].strip() != 'OFF':
        raise ValueError("No es un archivo OFF válido")

    parts = lines[1].strip().split()
    # Vértices indicados en el archivo
    num_vertices = int(parts[0])

    vertices = []
    # Los vértices aparecen a partir de la segunda linea
    for i in range(2, 2 + num_vertices):
        vertices.append([float(v) for v in lines[i].strip().split()])

    return np.array(vertices, dtype=np.float32)

# Función para guardar una imagen 3D de la nube de puntos
def save_3d_snapshot(points, save_path):
    fig = plt.figure()
    # Vista 3D
    ax = fig.add_subplot(111, projection='3d')
    # Puntos en color negro
    ax.scatter(points[:,0], points[:,1], points[:,2], s=1, c='black')
    # Desactivamos ejes
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

if len(sys.argv) != 2:
    print("Uso: python3 pointnet_predict_off.py ruta_archivo.off")
    sys.exit(1)

file_path = sys.argv[1]
points = load_off_points(file_path)

# Lista de clases del dataset ModelNet40
modelnet_classes = [
    "xbox", "wardrobe", "vase", "tv_stand", "toilet", "tent", "table", "stool", 
    "stairs", "sofa", "sink", "range_hood", "radio", "plant", "piano", "person", 
    "night_stand", "monitor", "mantel", "laptop", "lamp", "keyboard", "guitar", 
    "glass_box", "flower_pot", "dresser", "door", "desk", "curtain", "cup", 
    "cone", "chair", "car", "bowl", "bottle", "bookshelf", "bench", "bed", 
    "bathtub", "airplane"
]


# NORMALIZACIÓN: centramos la nube en el origen y escalamos a radio unidadpoints -= np.mean(points, axis=0)
points /= np.max(np.linalg.norm(points, axis=1))

if len(points) < 1024:
    indices = np.random.choice(len(points), 1024 - len(points), replace=True)
    points = np.concatenate([points, points[indices]])

# CARGA DEL MODELO PointNet + clasificador
model = Classifier(feature_model=PointNet(emb_dims=1024, use_bn=True))
model.load_state_dict(torch.load("./ruta_pointnet.t7", map_location="cpu"))
model.eval()

# INFERENCIA: se calcula la probabilidad de cada clase
with torch.no_grad():
    logits = model(torch.tensor(points).unsqueeze(0))  # Se añade dimensión de batch para procesar varios archivos
    probs = torch.softmax(logits, dim=1)               # Se convierte a probabilidad
    top5_probs, top5_indices = torch.topk(probs, 5)    # Se extraen las 5 clases más probables

# --- Mostrar resultados ---
print("\nTop 5 predicciones más probables:")
for i in range(5):
    class_idx = top5_indices[0][i].item()
    class_name = modelnet_classes[class_idx]
    confidence = top5_probs[0][i].item() * 100
    print(f"{i+1}. {class_name}: {confidence:.2f}%")

# Guardar snapshot
base_name = os.path.splitext(os.path.basename(file_path))[0]
img_path = f"{base_name}_snapshot.png"
save_3d_snapshot(points, img_path)
print(f"Snapshot guardado como {img_path}")
