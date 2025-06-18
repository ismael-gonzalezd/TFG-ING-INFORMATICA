#!/usr/bin/env python3
"""
===========================================================================
Nombre del archivo: pointcloud_saver.py

Descripción:
    Nodo de ROS que escucha un tópico de tipo PointCloud2, extrae las nubes 
    de puntos y las guarda en archivos PLY en una carpeta del usuario. Se utiliza 
    para registrar datos 3D de sensores RGB-D.

Funcionalidades:
    1. Suscripción a un tópico de ROS con mensajes PointCloud2.
    2. Conversión de los datos de ROS a arrays de NumPy.
    3. Guardado de cada nube de puntos como archivo PLY en formato ASCII.
    4. Control automático de nombres de archivo mediante un contador.

Requisitos:
    - Python 3.8 o superior
    - ROS (rospy, sensor_msgs)
    - NumPy

Uso:
    - Se puede lanzar el nodo y configurarlo mediante el parámetro `~topic`.
    - Las nubes se guardarán automáticamente en ~/Desktop/pointclouds/.

Autor: Ismael González Durán  
Fecha: 2025
===========================================================================
"""

import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
import os
import numpy as np
import getpass

class PointCloudSaver:
    def __init__(self):
        # Inicializa el nodo ROS
        rospy.init_node('pointcloud_saver', anonymous=True)

        # Obtiene la ruta al escritorio del usuario
        desktop_path = os.path.expanduser(f"~{getpass.getuser()}/Desktop")

        # Define la carpeta de salida donde se guardarán las nubes
        self.output_dir = os.path.join(desktop_path, "pointclouds")

        # Lee el nombre del tópico desde los parámetros de ROS o usa uno por defecto
        self.topic_name = rospy.get_param('~topic', '/robot/front_rgbd_camera/depth/points_throttle')

        # Contador para nombrar los archivos secuencialmente
        self.counter = 0

        # Crear la carpeta de salida si no existe
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Suscripción al tópico de PointCloud2
        rospy.Subscriber(self.topic_name, PointCloud2, self.callback)
        rospy.loginfo(f"Subscribed to {self.topic_name}, saving PLY files in {self.output_dir}")
        rospy.spin()

    def callback(self, msg):
        # Convertir el mensaje PointCloud2 a una lista de puntos (x, y, z)
        points = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
        if not points:
            rospy.logwarn("Nube vacía, se omite.")
            return

        # Convertir la lista de puntos a un array de NumPy
        points_np = np.array(points, dtype=np.float32)

        # Define el nombre del archivo donde guardar la nube de puntoss
        filename = os.path.join(self.output_dir, f"cloud_{self.counter:04d}.ply")

        # Guarda los puntos como archivo .ply
        self.save_as_ply(filename, points_np)
        rospy.loginfo(f"Nube de puntos guardada: {filename}")

        self.counter += 1

    def save_as_ply(self, filename, points):
        # Guarda un archivo PLY 
        with open(filename, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")
            for p in points:
                f.write(f"{p[0]} {p[1]} {p[2]}\n")

if __name__ == '__main__':
    try:
        PointCloudSaver()
    except rospy.ROSInterruptException:
        pass
