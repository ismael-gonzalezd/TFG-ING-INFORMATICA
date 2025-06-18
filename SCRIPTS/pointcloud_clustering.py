"""
===========================================================================
Nombre del archivo: pointcloud_clustering.py

Descripción:
    Este script carga una nube de puntos en formato .ply, la reduce mediante
    voxelización y aplica clustering con el algoritmo DBSCAN para
    segmentar los objetos presentes en la nube. Cada clúster se guarda 
    como un archivo independiente y se genera una nube de puntos coloreada 
    para visualizar los resultados.

Funcionalidades:
    - Carga de nubes de puntos .ply.
    - Voxelización para reducir la densidad de la nube.
    - Segmentación de clústeres usando DBSCAN.
    - Almacenamiento individual de cada clúster detectado.
    - Visualización y guardado de una nube de puntos coloreada.

Requisitos:
    - Python 3.8
    - Open3D
    - NumPy
    - Matplotlib

Uso:
    Ejecutar el script desde terminal:
        python3 pointcloud_clustering.py ruta_archivo.ply

    Donde:
        ruta_archivo.ply → es la ruta al archivo de nube de puntos a procesar.

Salida:
    - Carpeta con archivos .ply por cada clúster encontrado.
    - Archivo `cluster_colored.ply` con todos los clústeres coloreados.

Autor: Ismael González Durán
Fecha: 2025
===========================================================================
"""

import os
import sys
import open3d as o3d
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# -----------------------------------------
# Función para reducir la densidad de la nube de puntos usando voxelización.
# voxel_size define el tamaño del vóxel utilizado para agrupar puntos.
# -----------------------------------------
def voxel_downsample(pcd, voxel_size=0.02):
    return pcd.voxel_down_sample(voxel_size=voxel_size)

# -----------------------------------------
# Función para extraer clústeres utilizando el algoritmo DBSCAN.
# eps: distancia máxima entre puntos para ser considerados vecinos.
# min_points: número mínimo de puntos para formar un clúster.
# Devuelve una lista de clústeres y las etiquetas de cada punto.
# -----------------------------------------
def extract_clusters(pcd, eps=0.25, min_points=20):
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))
    max_label = labels.max()
    clusters = []
    for i in range(max_label + 1):
        indices = np.where(labels == i)[0]
        if len(indices) > 0:
            clusters.append(pcd.select_by_index(indices))
    return clusters, labels

# -----------------------------------------
# Guarda cada clúster en un archivo .ply separado en el directorio indicado.
# -----------------------------------
def save_clusters(clusters, base_dir):
    # Crear de la carpeta si no existe
    os.makedirs(base_dir, exist_ok=True)
    for idx, cluster in enumerate(clusters):
        filename = os.path.join(base_dir, f"cluster_{idx:03d}.ply")
        o3d.io.write_point_cloud(filename, cluster)
        print(f"Cluster guardado: {filename}")

# -----------------------------------------
# Guarda una nube combinada donde cada clúster calculado se muestra con un color diferente.
# -----------------------------------------
def visualize_clusters(pcd_down, labels):
    max_label = labels.max()
    # Calcular colores diferentes para cada clúster usando la paleta de colores tab20
    colors = plt.get_cmap("tab20")(labels / (max_label + 1 if max_label > 0 else 1))
    colors[labels < 0] = [0, 0, 0, 1] 
    pcd_down.colors = o3d.utility.Vector3dVector(colors[:, :3])  # Asignar colores a la nube
    # Guardar nube coloreada
    o3d.io.write_point_cloud("cluster_colored.ply", pcd_down)
    print("Nube coloreada guardada como cluster_colored.ply")

# -----------------------------------------
# Función principal para procesar el archivo .ply: Carga, reduce, segmenta, guarda y visualiza los clústeres.
# -----------------------------------------
def process_ply_file(ply_path):
    if not os.path.isfile(ply_path):
        print(f"Archivo no encontrado: {ply_path}")
        return

    # Crear nombre de carpeta de salida basado en el nombre del archivo .ply
    base_name = os.path.splitext(os.path.basename(ply_path))[0]
    output_dir = os.path.join(os.path.dirname(ply_path), base_name + "_clusters")

    # Cargar nube de puntos
    pcd = o3d.io.read_point_cloud(ply_path)
    print("Nube original cargada")

    # Reducir densidad de puntos mediante voxelización
    pcd_down = voxel_downsample(pcd)
    print("Nube voxelizada")

    # Extraer clústeres usando el algoritmo DBSCAN
    clusters, labels = extract_clusters(pcd_down)
    print(f"{len(clusters)} clusters encontrados")

    # Guardar cada clúster en archivos individuales
    save_clusters(clusters, output_dir)
    print("Proceso completado")

    # Visualizar y guardar nube coloreada con los clústeres
    visualize_clusters(pcd_down, labels)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python3 script.py path_a_nube.ply")
        sys.exit(1)

    ply_file = sys.argv[1]
    process_ply_file(ply_file)

