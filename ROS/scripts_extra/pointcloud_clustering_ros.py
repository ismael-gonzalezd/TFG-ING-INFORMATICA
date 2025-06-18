#!/usr/bin/env python3
"""
===========================================================================
Nombre del archivo: pointcloud_clustering_ros.py

Descripción:
    Adaptación a ROS del script para voxelizar y segmentar una nube de puntos en clusteres. 
    En lugar de guardarlos como archivos .ply independientes, se envían a un tópico llamado 
    segmented_cloud

Funcionalidades:
    1. Escucha una nube de puntos desde un sensor RGB-D.
    2. Reduce la densidad de la nube con voxelizacion.
    3. Aplica clustering con DBSCAN para identificar objetos o regiones.
    4. Publica cada cluster individualmente en `/segmented_cloud`.
    5. Visualiza los clusters con colores distintos en Open3D.

Requisitos:
    - Python 3.8+
    - ROS (rospy, sensor_msgs)
    - NumPy
    - Open3D
    - scikit-learn
    - Matplotlib

Uso:
    - Lanza este nodo con una nube publicada en `/robot/front_rgbd_camera/depth/points`.
    - Recibe los clusters en `/segmented_cloud` y visualiza en 3D.

Autor: Ismael González Durán  
Fecha: 2025
===========================================================================
"""

import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
import numpy as np
from sklearn.cluster import DBSCAN
import open3d as o3d
import matplotlib.pyplot as plt

class ClusterPublisher:
    def __init__(self):
        rospy.init_node("cluster_publisher", anonymous=True)

        # Suscribirse a la nube de puntos principal
        self.pc_sub = rospy.Subscriber("/robot/front_rgbd_camera/depth/points", PointCloud2, self.pc_callback)

        # Publicador para enviar los clusters detectados
        self.cluster_pub = rospy.Publisher("/detected_clusters", PointCloud2, queue_size=10)

        # Controlar frecuencia de procesamiento
        self.rate = rospy.Rate(10)  # 10 Hz

    def pc_callback(self, msg):
        # Extraer puntos 3D desde PointCloud2
        points = np.array([p[:3] for p in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)])

        if len(points) == 0:
            return

        # Convertir a nube Open3D y aplicar una rotación para corregir eje (si necesario)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        R = np.array([[1,  0,  0],  
                      [0, -1,  0],  
                      [0,  0, -1]])  # Rotación para sistema ROS

        pcd.rotate(R, center=(0, 0, 0))

        # Voxelización para reducir cantidad de puntos
        voxel_size = 0.02
        downsampled_pcd = pcd.voxel_down_sample(voxel_size)
        downsampled_points = np.asarray(downsampled_pcd.points)

        # Clustering con DBSCAN
        dbscan = DBSCAN(eps=0.15, min_samples50)
        labels = dbscan.fit_predict(downsampled_points)

        unique_labels = np.unique(labels)
        cluster_pcds = []

        # Visualización con colores distintos para cada cluster
        cmap = plt.get_cmap("tab20")
        cluster_colors = cmap((labels % 20) / 20)[:, :3]

        for idx, label in enumerate(unique_labels):
            if label == -1:
                continue  # Ignorar ruido

            # Extraer puntos del cluster actual
            cluster_points = downsampled_points[labels == label]
            num_points = len(cluster_points)

            rospy.loginfo(f"Cluster {label}: {num_points} puntos")

            # Crear objeto de visualización para Open3D
            cluster_pcd = o3d.geometry.PointCloud()
            cluster_pcd.points = o3d.utility.Vector3dVector(cluster_points)
            color = cmap(idx)[:3]
            cluster_pcd.paint_uniform_color(color)
            cluster_pcds.append(cluster_pcd)

            # Crear mensaje PointCloud2 para publicar el clúster
            header = msg.header
            fields = [
                PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1)
            ]
            cluster_msg = pc2.create_cloud(header, fields, cluster_points)

            # Publicar el cluster
            self.cluster_pub.publish(cluster_msg)

        # Visualizar todos los clusters juntos
        o3d.visualization.draw_geometries(cluster_pcds)
        rospy.loginfo("Número de clusters detectados: " + str(len(cluster_pcds)))

        self.rate.sleep()

if __name__ == "__main__":
    try:
        ClusterPublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
