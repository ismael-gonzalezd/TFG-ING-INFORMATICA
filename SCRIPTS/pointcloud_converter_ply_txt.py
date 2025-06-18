#!/usr/bin/env python3
"""
===========================================================================
Nombre del archivo: convertir_pcl_a_txt.py

Descripción:
    Este script carga una nube de puntos desde un archivo PCL (formato .pcd o .ply),
    extrae las coordenadas XYZ y las guarda como un archivo .txt con formato plano.

Uso:
    python3 convertir_pcl_a_txt.py ruta/a/archivo.pcd
    python3 convertir_pcl_a_txt.py ruta/a/archivo.ply

Salida:
    - nube_convertida.txt (en el mismo directorio donde se ejecuta el script)

Dependencias:
    - Open3D
    - NumPy

Autor: Ismael González Durán
Fecha: 2025
===========================================================================
"""

import sys
import os
import open3d as o3d
import numpy as np

# --- Comprobación de argumentos ---
if len(sys.argv) < 2:
    print("Uso: python3 convertir_pcl_a_txt.py archivo.pcd/.ply")
    sys.exit(1)

ruta_pcl = sys.argv[1]

# --- Carga de la nube de puntos ---
try:
    pcd = o3d.io.read_point_cloud(ruta_pcl)
    if not pcd.has_points():
        raise ValueError("La nube de puntos está vacía o no se pudo cargar.")
except Exception as e:
    print(f"Error al cargar la nube de puntos: {e}")
    sys.exit(1)

# --- Conversión a array numpy (X, Y, Z) ---
points = np.asarray(pcd.points)

# --- Guardar como .txt ---
nombre_base = os.path.splitext(os.path.basename(ruta_pcl))[0]
ruta_salida = os.path.join(os.getcwd(), f"{nombre_base}.txt")

try:
    np.savetxt(ruta_salida, points, fmt="%.6f")
    print(f"Nube convertida guardada en: {ruta_salida}")
except Exception as e:
    print(f"No se pudo guardar el archivo .txt: {e}")
