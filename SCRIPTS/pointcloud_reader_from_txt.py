"""
===========================================================================
Nombre del archivo: pointcloud_reader_from_txt.py

Descripción:
    Script para cargar y visualizar una nube de puntos 3D almacenada
    en un archivo de texto. Usa Open3D para mostrarla en una ventana interactiva
    (hace falta un entorno gráfico OpenGL).

    Alternativa: Instalar programa Meshlab para abrir la nube PLY

Requisitos:
    - NumPy
    - Open3D
    - Archivo de entrada en formato de texto (.txt) con una nube de puntos
      (una fila por punto: X Y Z separados por espacios)

Uso:
    python3 visualizar_nube_open3d.py ruta/a/nube.txt [--color R G B] [--size TAMAÑO]

    Ejemplos:
    - Visualización por defecto:
        python3 visualizar_nube_open3d.py nube.txt
    - Visualización en rojo:
        python3 visualizar_nube_open3d.py nube.txt --color 1 0 0
    - Visualización con puntos grandes:
        python3 visualizar_nube_open3d.py nube.txt --size 5.0

Autor: Ismael González Durán 
Fecha: 2025
===========================================================================
"""
import numpy as np
import open3d as o3d
import argparse

# --- Argumentos de entrada ---
parser = argparse.ArgumentParser(description="Visualiza una nube de puntos 3D con Open3D.")
parser.add_argument("path", help="Ruta al archivo de la nube de puntos (formato .txt con columnas X Y Z)")
parser.add_argument("--color", type=float, nargs=3, metavar=('R', 'G', 'B'), default=None,
                    help="Color RGB para los puntos (valores entre 0 y 1)")
parser.add_argument("--size", type=float, default=1.0,
                    help="Tamaño de los puntos (default: 1.0)")
args = parser.parse_args()

# --- Carga la nube de puntos ---
try:
    points = np.loadtxt(args.path)
    if points.shape[1] != 3:
        raise ValueError("Error de formato: El archivo debe contener exactamente 3 columnas (X Y Z)")
except Exception as e:
    print(f"Error al cargar la nube de puntos: {e}")
    exit()

# --- Crea y configura la nube de puntos ---
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# Si se especifica un color, se aplica a todos los puntos
if args.color:
    color = np.array(args.color)
    if np.any(color < 0) or np.any(color > 1):
        print("Los valores de color RGB deben estar entre 0 y 1.")
        exit(1)
    pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (points.shape[0], 1)))

# --- Visualización con configuración personalizada ---
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Visualización de nube de puntos", width=800, height=600)
vis.add_geometry(pcd)

# Ajustar tamaño de los puntos
render_option = vis.get_render_option()
render_option.point_size = args.size

# Ejecutar visualización
vis.run()
vis.destroy_window()
