"""
===========================================================================
Nombre del archivo: registro_homografia.py

Descripción:
    Este script permite calcular la matriz de homografía con la que proyectar
    los puntos de una imagen térmica sobre el plano RGB de manera correcta. 
    El usuario selecciona puntos manualmente en ambas imágenes, se calcula 
    la homografía entre ellas y se verifica visualmente el alineamiento 
    mediante la proyección de los puntos térmicos sobre la imagen RGB.

Funcionalidades:
    1. Carga y visualización de imágenes térmicas y RGB.
    2. Selección manual de puntos correspondientes entre ambas imágenes.
    3. Cálculo de la matriz de homografía con el método RANSAC.
    4. Visualización comparativa de puntos originales y proyectados.
    5. Cálculo del error medio de proyección.

Requisitos:
    - Python 3.8 o superior
    - OpenCV (cv2)
    - NumPy

Uso:
    - Ejecuta el script.
    - Selecciona al menos 4 puntos en la imagen térmica.
    - Luego selecciona los puntos correspondientes en la imagen RGB.
    - El script calculará la homografía y mostrará los resultados.

Autor: Ismael González Durán
Fecha: 2025
===========================================================================
"""

import cv2
import os
import numpy as np

# --- 1. Selección de puntos ---
def seleccionar_puntos(imagen_path, window_name):
    # Cargar imagen desde el path pasado por argumentos
    imagen = cv2.imread(imagen_path)
    if imagen is None: 
        print(f"No se pudo cargar la imagen: {imagen_path}")
        return []
    
    imagen_mostrar = imagen.copy()
    puntos = []

    # Función callback para registrar los puntos que se graban al hacer click en la imagen
    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            puntos.append((x, y))
            # Dibujar circulo y punto sobre la imagen al hacer click
            cv2.circle(imagen_mostrar, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(imagen_mostrar, str(len(puntos)), (x+10, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.imshow(window_name, imagen_mostrar)
    
    # Mostrar imagen y activar la función de registrar los clicks del ratón
    cv2.imshow(window_name, imagen_mostrar)
    cv2.setMouseCallback(window_name, click_event)
    print(f"[{window_name}] Selecciona puntos correspondientes (mínimo 4, recomendado 8-10).")
    print("Pulsa 'r' para reiniciar, cualquier otra tecla para continuar.")
    
    while True:
        # Esperar a que el usuario vuelva a pulsar otro punto, continuar o reiniciar (pulsando R)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('r'):  # Reiniciar selección de puntos (implica borrar los que ya están registrados)
            puntos = []
            imagen_mostrar = imagen.copy()
            cv2.imshow(window_name, imagen_mostrar)
        else:
            break
    
    cv2.destroyAllWindows()
    return np.float32(puntos)

# --- 2. Verificación visual ---
def verificar_homografia(imagen_thermal_path, imagen_rgb_path, H, pts_thermal, pts_rgb):
    # Cargar imágenes térmica y RGB
    thermal = cv2.imread(imagen_thermal_path)
    rgb = cv2.imread(imagen_rgb_path)
    
    # Proyectar puntos térmicos a RGB usando la matriz de homografía calculada
    pts_thermal_projected = cv2.perspectiveTransform(pts_thermal.reshape(-1, 1, 2), H)
    pts_thermal_projected = pts_thermal_projected.reshape(-1, 2)
    
    # Crear imagen de verificación (poner ambas imágenes lado a lado)
    h_thermal, w_thermal = thermal.shape[:2]
    h_rgb, w_rgb = rgb.shape[:2]
    max_h = max(h_thermal, h_rgb)
    combined = np.zeros((max_h, w_thermal + w_rgb, 3), dtype=np.uint8)
    combined[:h_thermal, :w_thermal] = thermal
    combined[:h_rgb, w_thermal:w_thermal+w_rgb] = rgb
    
    # Dibujar líneas y puntos correspondientes
    for i, (pt_thermal, pt_rgb, pt_projected) in enumerate(zip(pts_thermal, pts_rgb, pts_thermal_projected)):
        # Color para este punto (HSV a BGR para colores distintos)
        color = tuple(map(int, np.array([i*180/len(pts_thermal), 255, 255])))
        color = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_HSV2BGR)[0][0].tolist()
        
        # Punto en imagen térmica (lado izquierdo)
        pt_thermal_int = tuple(map(int, pt_thermal))
        cv2.circle(combined, pt_thermal_int, 8, color, -1)
        cv2.putText(combined, str(i+1), (pt_thermal_int[0]+10, pt_thermal_int[1]-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        # Punto seleccionado manualmente en RGB (lado derecho)
        pt_rgb_int = tuple(map(int, pt_rgb))
        pt_rgb_combined = (pt_rgb_int[0] + w_thermal, pt_rgb_int[1])
        cv2.circle(combined, pt_rgb_combined, 8, color, -1)
        cv2.putText(combined, str(i+1), (pt_rgb_combined[0]+10, pt_rgb_combined[1]-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        # Punto térmico proyectado por homografía en RGB (lado derecho)
        pt_projected_int = tuple(map(int, pt_projected))
        pt_projected_combined = (pt_projected_int[0] + w_thermal, pt_projected_int[1])
        cv2.circle(combined, pt_projected_combined, 8, color, 2)
        
        # Línea conectando punto manual y proyectado
        cv2.line(combined, pt_rgb_combined, pt_projected_combined, color, 2)
        
        # Calcular error para este punto
        error = np.linalg.norm(pt_rgb - pt_projected)
        print(f"Punto {i+1}: Error = {error:.2f} píxeles")
    
    # Calcular error medio de todos los puntos
    errors = np.linalg.norm(pts_rgb - pts_thermal_projected, axis=1)
    mean_error = np.mean(errors)
    print(f"\nError medio: {mean_error:.2f} píxeles")
    
    # Mostrar imagen de verificación con el resultado final
    cv2.putText(combined, f"Error medio: {mean_error:.2f} px", (20, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imshow("Verificacion Homografia - Circulos: puntos manuales, Contornos: puntos proyectados", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return mean_error

# --- 3. Main ---
def main():
    # Rutas dde las imágenes (sustituir por el path de las imágenes cuya homografía se quiere calcular)
    ruta_termica = os.path.expanduser("./path_termico")
    ruta_rgb = os.path.expanduser("./path_rgb")

    # Paso 1: seleccionar puntos en ambas imágenes
    print("=== Selección de puntos en imagen térmica ===")
    pts_thermal = seleccionar_puntos(ruta_termica, "Thermal")
    print("\n=== Selección de puntos en imagen RGB ===")
    pts_rgb = seleccionar_puntos(ruta_rgb, "RGB")

    # Verificar que se usan al menos 4 puntos y que haya la misma cantidad de puntos en cada imagen
    if len(pts_thermal) < 4 or len(pts_rgb) < 4:
        print("Necesitas al menos 4 puntos en cada imagen")
        return
    elif len(pts_thermal) != len(pts_rgb):
        print("Debes seleccionar el mismo número de puntos en ambas imágenes")
        return

    # Paso 2: calcular homografía entre imagen térmica y RGB
    H, status = cv2.findHomography(pts_thermal, pts_rgb, cv2.RANSAC)
    print("\nMatriz de homografía H:")
    print(H)

    # Paso 3: verificar visualmente la homografía calculada
    print("\n=== Verificación de la homografía ===")
    mean_error = verificar_homografia(ruta_termica, ruta_rgb, H, pts_thermal, pts_rgb)
    
    # Evaluación de resultados basada en error medio
    if mean_error < 5.0:
        print("\nHomografía precisa (error < 5 px)")
    elif mean_error < 10.0:
        print("\nHomografía aceptable (error 5-10 px)")
    else:
        print("\nHomografía imprecisa (error > 10 px)")

if __name__ == "__main__":
    main()
