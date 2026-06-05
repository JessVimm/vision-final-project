import numpy as np
import trimesh
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import os
import tkinter as tk
from tkinter import filedialog
from collections import deque
from scipy.ndimage import binary_dilation, binary_closing
from matplotlib.animation import FuncAnimation, PillowWriter

# 1. CARGA Y PREPROCESAMIENTO 
def load_and_clean_mesh(filepath):
    """Carga puntos y los limpia."""
    mesh = trimesh.load(filepath)
    points = mesh.vertices if hasattr(mesh, 'vertices') else mesh
    points = np.array(points)
    points = points[~np.isnan(points).any(axis=1)]
    points = points[~np.isinf(points).any(axis=1)]
    return points

def align_object_with_inertia_tensor(points):
    """Alinea el objeto usando sus ejes principales."""
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    I = np.zeros((3, 3))
    for x, y, z in points_centered:
        I[0, 0] += y**2 + z**2
        I[1, 1] += x**2 + z**2
        I[2, 2] += x**2 + y**2
        I[0, 1] -= x * y
        I[0, 2] -= x * z
        I[1, 2] -= y * z
    I[1, 0] = I[0, 1]; I[2, 0] = I[0, 2]; I[2, 1] = I[1, 2]
    eigvals, eigvecs = eigh(I)
    R = eigvecs 
    points_aligned = points_centered @ R
    return points_aligned, points_centered, centroid, I, eigvals, eigvecs

def fill_interior(volume):
    visited = np.zeros_like(volume, dtype=bool)
    queue = deque()
    nx, ny, nz = volume.shape
    for x in [0, nx-1]:
        for y in range(ny):
            for z in range(nz):
                if volume[x,y,z] == 0:
                    visited[x,y,z] = True
                    queue.append((x,y,z))
    for y in [0, ny-1]:
        for x in range(nx):
            for z in range(nz):
                if volume[x,y,z] == 0 and not visited[x,y,z]:
                    visited[x,y,z] = True
                    queue.append((x,y,z))
    for z in [0, nz-1]:
        for x in range(nx):
            for y in range(ny):
                if volume[x,y,z] == 0 and not visited[x,y,z]:
                    visited[x,y,z] = True
                    queue.append((x,y,z))
    directions = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
    while queue:
        x,y,z = queue.popleft()
        for dx,dy,dz in directions:
            xx, yy, zz = x + dx, y + dy, z + dz
            if 0 <= xx < nx and 0 <= yy < ny and 0 <= zz < nz:
                if volume[xx,yy,zz] == 0 and not visited[xx,yy,zz]:
                    visited[xx,yy,zz] = True
                    queue.append((xx,yy,zz))
    volume_filled = volume.copy()
    volume_filled[(volume == 0) & (~visited)] = 1
    return volume_filled

def build_filled_voxels(points, voxel_size):
    grid = np.floor(points / voxel_size).astype(int)
    grid -= grid.min(axis=0)
    shape = grid.max(axis=0) + 5
    volume = np.zeros(shape, dtype=np.uint8)
    volume[grid[:,0], grid[:,1], grid[:,2]] = 1
    volume = binary_dilation(volume, iterations=1)
    volume = binary_closing(volume, iterations=1)
    volume = fill_interior(volume)
    voxels = np.argwhere(volume > 0).astype(float)
    voxels -= np.mean(voxels, axis=0)
    return voxels


# 2. DISCRETIZACIÓN
def quantize_to_target_volume(points, target_n, max_iters=500):
    points_centered = points - np.mean(points, axis=0)
    bbox = points_centered.max(axis=0) - points_centered.min(axis=0)
    max_dim = np.max(bbox)
    low_div, high_div = 5.0, 300.0
    best_voxels, best_diff, best_v_size = None, float('inf'), 1.0
    for _ in range(max_iters):
        mid_div = (low_div + high_div) / 2.0
        v_size = max_dim / mid_div
        voxels_idx = np.unique(np.floor(points_centered / v_size).astype(int), axis=0)
        current_n = len(voxels_idx)
        if abs(current_n - target_n) < best_diff:
            best_diff = abs(current_n - target_n)
            best_voxels = voxels_idx
            best_v_size = v_size
        if current_n == target_n: break
        elif current_n < target_n: low_div = mid_div 
        else: high_div = mid_div 
    
    current_voxels = list(best_voxels)
    if len(current_voxels) > target_n:
        dists = np.linalg.norm(current_voxels, axis=1)
        best_voxels = np.array(current_voxels)[np.argsort(dists)[:target_n]]
    elif len(current_voxels) < target_n:
        needed = target_n - len(current_voxels)
        voxel_set = set(tuple(v) for v in current_voxels)
        candidates = set()
        for v in current_voxels:
            for d in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
                cand = (v[0]+d[0], v[1]+d[1], v[2]+d[2])
                if cand not in voxel_set: candidates.add(cand)
        candidates_arr = np.array(list(candidates))
        closest = candidates_arr[np.argsort(np.linalg.norm(candidates_arr, axis=1))[:needed]]
        best_voxels = np.vstack((best_voxels, closest))
    return best_voxels.astype(float), best_v_size

def get_voxel_differences(points_A, points_B):
    set_A, set_B = set(tuple(p) for p in points_A), set(tuple(p) for p in points_B)
    common = set_A.intersection(set_B)
    return (np.array(list(set_A - common)) if set_A - common else np.empty((0,3)), 
            np.array(list(set_B - common)) if set_B - common else np.empty((0,3)), 
            np.array(list(common)) if common else np.empty((0,3)))


# 3. ALGORITMO HÚNGARO
def optimal_transformation(pos_voxels, neg_voxels):
    if len(pos_voxels) == 0 or len(neg_voxels) == 0: return 0.0, [], [], []
    C = cdist(pos_voxels, neg_voxels, metric='euclidean')
    row_ind, col_ind = linear_sum_assignment(C)
    return C[row_ind, col_ind].sum(), C[row_ind, col_ind], row_ind, col_ind


# 4. EXPORTACIÓN
def export_summary_report_txt(filename, name1, name2, voxel_size, pos_len, neg_len, com_len, total_work, distances, row_ind, col_ind, cent1, cent2, tensor1, tensor2, eval1, eval2, evec1, evec2, total_pts1, total_pts2):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("========== RESULTADOS DE SIMILITUD ==========\n\n")
        f.write(f"Objeto 1: {name1}\nObjeto 2: {name2}\n")
        f.write(f"Tamaño de Voxel (Voxel size): {voxel_size:.6f}\n\n")
        
        f.write("========== DATOS GEOMÉTRICOS ==========\n\n")
        
        f.write(f"Voxeles {name1}: {total_pts1}\n")
        f.write(f"Voxeles {name2}: {total_pts2}\n\n")
        
        f.write(f"--- {name1} ---\n")
        f.write(f"Centroide: {np.round(cent1, 4)}\n")
        f.write(f"Tensor de Inercia:\n{np.round(tensor1, 4)}\n")
        f.write(f"Eigenvalores: {np.round(eval1, 4)}\n")
        f.write(f"Eigenvectores:\n{np.round(evec1, 4)}\n\n")

        f.write(f"--- {name2} ---\n")
        f.write(f"Centroide: {np.round(cent2, 4)}\n")
        f.write(f"Tensor de Inercia:\n{np.round(tensor2, 4)}\n")
        f.write(f"Eigenvalores: {np.round(eval2, 4)}\n")
        f.write(f"Eigenvectores:\n{np.round(evec2, 4)}\n\n")

        f.write("========== DIFERENCIA DE VOXELES ==========\n\n")
        f.write(f"Voxeles Comunes (Intersección): {com_len}\n")
        f.write(f"Sobrantes (Positivos): {pos_len}\n")
        f.write(f"Huecos (Negativos): {neg_len}\n\n")
        
        f.write("========== TRANSPORTE ÓPTIMO ==========\n\n")
        f.write(f"Costo de Trabajo total: {total_work:.4f}\n\n")
        if len(distances) > 0:
            f.write(f"Distancia mínima: {np.min(distances):.4f}\n")
            f.write(f"Distancia máxima: {np.max(distances):.4f}\n")
            f.write(f"Distancia promedio: {np.mean(distances):.4f}\n")
            f.write(f"Desviación estándar: {np.std(distances):.4f}\n")
        else:
            f.write("Distancia mínima: 0.0\nDistancia máxima: 0.0\nDistancia promedio: 0.0\nDesviación estándar: 0.0\n")
            
        f.write("\n========== ASIGNACIONES (Mapeo Húngaro) ==========\n\n")
        for p_idx, q_idx, dist in zip(row_ind, col_ind, distances):
            f.write(f"Sobrante P[{p_idx}] -> Hueco Q[{q_idx}] | Distancia = {dist:.4f}\n")

def export_result_as_voxel_mesh(filename, final_points, voxel_size):
    if len(final_points) == 0:
        return
    print(f"Exportando {filename} ...")
    min_p = np.min(final_points, axis=0)
    indices = np.round(final_points - min_p).astype(int)
    shape = np.max(indices, axis=0) + 1
    matrix = np.zeros(shape, dtype=bool)
    matrix[indices[:, 0], indices[:, 1], indices[:, 2]] = True
    mesh = trimesh.voxel.VoxelGrid(matrix).as_boxes()
    mesh.apply_scale(voxel_size)
    mesh.apply_translation(min_p * voxel_size)
    mesh.export(filename)

def build_transported_object(quantized1, common_voxels, pos_voxels, neg_voxels, rows, cols):
    final_volume = set(tuple(v) for v in quantized1)
    movimientos = []
    for i, j in zip(rows, cols):
        origen, destino = tuple(pos_voxels[i]), tuple(neg_voxels[j])
        if origen in final_volume: final_volume.remove(origen)
        final_volume.add(destino)
        movimientos.append((origen, destino))
    return np.array(list(final_volume), dtype=float), movimientos

# 5. VISUALIZACIÓN Y PRINCIPAL
def visualize_alignment(orig, align, name):
    fig = plt.figure(figsize=(12, 6))
    for i, (pts, tit) in enumerate([(orig, "Original"), (align, "Alineado")]):
        ax = fig.add_subplot(1, 2, i+1, projection='3d')
        ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=1, c='black')
        ax.set_title(f"{name} - {tit}")
    plt.show()

def plot_pipeline_object(orig, cent, align, vox, name):
    fig = plt.figure(figsize=(18,5))
    for i, (pts, tit) in enumerate([(orig, "Original"), (cent, "Centrado"), (align, "Alineado"), (vox, f"Voxelizado ({len(vox)})")]):
        ax = fig.add_subplot(1, 4, i + 1, projection='3d')
        ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=3)
        ax.set_title(tit)
    plt.show()

def plot_alignment(points_centered, points_aligned, name):
    """Grafica el objeto original centrado vs el objeto alineado."""
    fig = plt.figure(figsize=(12, 6))
    
    max_puntos = 3500
    step = max(1, len(points_centered) // max_puntos)
    
    pts_c_vis = points_centered[::step]
    pts_a_vis = points_aligned[::step]
    
    # Original
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(pts_c_vis[:, 0], pts_c_vis[:, 1], pts_c_vis[:, 2], 
                s=1, c='black', alpha=0.6)
    ax1.set_title(f"{name} - Original")
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # Alineado
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(pts_a_vis[:, 0], pts_a_vis[:, 1], pts_a_vis[:, 2], 
                s=1, c='black', alpha=0.6)
    ax2.set_title(f"{name} - Alineado")
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.zaxis.labelpad = -5
    
    plt.show()

def plot_voxel_differences(pos_voxels, neg_voxels, com_voxels, name1, name2):
    """Grafica los sobrantes, huecos y comunes."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    max_puntos = 3500
    
    def subsample_random(voxels):
        if len(voxels) <= max_puntos:
            return voxels
        indices = np.random.choice(len(voxels), max_puntos, replace=False)
        return voxels[indices]
        
    com_vis = subsample_random(com_voxels)
    pos_vis = subsample_random(pos_voxels)
    neg_vis = subsample_random(neg_voxels)
    
    # Comunes (Verde)
    if len(com_vis) > 0:
        ax.scatter(com_vis[:, 0], com_vis[:, 1], com_vis[:, 2], 
                   s=2, c='green', label='Comunes', alpha=0.2)
    
    # Sobrantes (Rojo)
    if len(pos_vis) > 0:
        ax.scatter(pos_vis[:, 0], pos_vis[:, 1], pos_vis[:, 2], 
                   s=5, c='red', label=f'Sobrantes ({name1})', alpha=0.6)
        
    # Huecos (Azul)
    if len(neg_vis) > 0:
        ax.scatter(neg_vis[:, 0], neg_vis[:, 1], neg_vis[:, 2], 
                   s=5, c='blue', label=f'Huecos ({name2})', alpha=0.6)

    ax.set_title("Comparación")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([1,1,1]) 
    ax.legend()
    plt.show()

    # GENERADOR DE FRAMES (Interpolación lineal)
def generate_transport_frames(pos_voxels, neg_voxels, com_voxels, rows, cols, num_frames=30):
    """Calcula las posiciones intermedias para la animación del transporte."""
    frames = []
    
    origenes = pos_voxels[rows]
    destinos = neg_voxels[cols]
    
    for t in np.linspace(0, 1, num_frames):
        # Fórmula de interpolación: (1-t)*A + t*B
        posiciones_actuales = origenes * (1 - t) + destinos * t
        
        # Juntamos los que se mueven con los comunes 
        if len(com_voxels) > 0:
            frame_completo = np.vstack((com_voxels, posiciones_actuales))
        else:
            frame_completo = posiciones_actuales
            
        frames.append(frame_completo)
        
    return frames

# FUNCIÓN DE ANIMACIÓN 
def animate_optimal_transport(frames, name1, name2):
    """Anima los frames generados con estética de nube de puntos."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # submuestreo 
    max_puntos = 3500
    step = max(1, len(frames[0]) // max_puntos)
    
    pts0 = frames[0][::step]
    
    # Graficamos el estado inicial 
    scatter = ax.scatter(pts0[:,0], pts0[:,1], pts0[:,2], s=5, c='purple', alpha=0.5)
    
    # Ajustamos los límites de la cámara viendo el primer y último frame
    all_pts = np.vstack((frames[0][::step], frames[-1][::step]))
    ax.set_xlim(all_pts[:,0].min(), all_pts[:,0].max())
    ax.set_ylim(all_pts[:,1].min(), all_pts[:,1].max())
    ax.set_zlim(all_pts[:,2].min(), all_pts[:,2].max())
    ax.set_box_aspect([1,1,1]) # Proporciones cúbicas

    def update(i):
        pts = frames[i][::step]
        scatter._offsets3d = (pts[:,0], pts[:,1], pts[:,2])
        ax.set_title(f"Transformación: {name1} -> {name2} | Frame {i+1}/{len(frames)}")
        return scatter,

    anim = FuncAnimation(fig, update, frames=len(frames), interval=60, blit=False)
    # Guardamos la animación
    save_animation(anim, name1, name2)
    
    plt.show()

def save_animation(anim, name1, name2):
    """Guarda la animación (GIF) y la última imagen (PNG) """
    # 1. Guardar Animación GIF
    gif_name = f"animacion_{name1}_a_{name2}.gif"
    writer = PillowWriter(fps=15)
    print(f"Guardando animación en {gif_name}...")
    anim.save(gif_name, writer=writer)
    print(f"Animación guardada como {gif_name}!")

def save_as_obj(filename, points):
    """Guarda los puntos usando el manejador nativo de trimesh."""
    pc = trimesh.PointCloud(points)
    if filename.endswith('.obj'):
        filename = filename.replace('.obj', '.ply')
        
    pc.export(filename)
    print(f"Archivo guardado correctamente: {filename}")

def run_shape_similarity(obj1_path, obj2_path, name1, name2, target_voxels=15000):
    print(f" INICIANDO PROCESO: {name1} vs {name2} ")
    
    print("\n Cargando objetos y limpiando mallas...")
    pts1, pts2 = load_and_clean_mesh(obj1_path), load_and_clean_mesh(obj2_path)
    
    print(" Alineando objetos...")
    pts1_a, pts1_c, cent1, tensor1, eval1, evec1 = align_object_with_inertia_tensor(pts1)
    pts2_a, pts2_c, cent2, tensor2, eval2, evec2 = align_object_with_inertia_tensor(pts2)

    plot_alignment(pts1_c, pts1_a, name1)
    plot_alignment(pts2_c, pts2_a, name2)
    
    print(" Voxelizando (espere un momento)...")
    bbox1, bbox2 = pts1_a.max(axis=0)-pts1_a.min(axis=0), pts2_a.max(axis=0)-pts2_a.min(axis=0)
    filled1 = build_filled_voxels(pts1_a, np.max(bbox1)/100)
    filled2 = build_filled_voxels(pts2_a, np.max(bbox2)/100)
    
    q1, v1 = quantize_to_target_volume(filled1, target_voxels)
    q2, v2 = quantize_to_target_volume(filled2, target_voxels)
    
    pos, neg, com = get_voxel_differences(q1, q2)
    plot_voxel_differences(pos, neg, com, name1, name2)
    
    print(f"Calculando Algoritmo Húngaro (esto puede tardar unos minutos)...")
    work, dists, rows, cols = optimal_transformation(pos, neg)

    print(" Generando animación...")
    frames_animacion = generate_transport_frames(pos, neg, com, rows, cols, num_frames=40)
    animate_optimal_transport(frames_animacion, name1, name2)
    
    # Exportar reporte TXT
    export_summary_report_txt(
        f"reporte_{name1}_vs_{name2}.txt", 
        name1, name2, (v1+v2)/2.0, len(pos), len(neg), len(com), 
        work, dists, rows, cols, 
        cent1, cent2, tensor1, tensor2, eval1, eval2, evec1, evec2, len(pts1), len(pts2)
    )
    
    # EXPORTACIÓN DE ARCHIVOS .OBJ 
    print(" Guardando archivos .obj...")
    
    #Guardar los normalizados 
    save_as_obj(f"{name1}_normalized.obj", pts1_c)
    save_as_obj(f"{name2}_normalized.obj", pts2_c)

    #Guardar el intermedio (Voxelizado)
    frame_mitad = frames_animacion[len(frames_animacion)//2]
    export_result_as_voxel_mesh(
        f"med_{name1}_vs_{name2}.obj", 
        frame_mitad, 
        (v1 + v2) / 2.0
    )

    #Guardar el final (Voxelizado)
    export_result_as_voxel_mesh(
        f"final_{name1}_vs_{name2}.obj", 
        frames_animacion[-1], 
        (v1 + v2) / 2.0
    )
    
    print("\nProceso completado con éxito Revisa los archivos generados.")

if __name__ == "__main__":
    root = tk.Tk(); root.withdraw()
    print("Seleccionando archivos...")
    p1 = filedialog.askopenfilename(title="Selecciona el PRIMER objeto (.obj)")
    if p1: print(f"Archivo 1 seleccionado: {os.path.basename(p1)}")
    p2 = filedialog.askopenfilename(title="Selecciona el SEGUNDO objeto (.obj)")
    if p2: print(f"Archivo 2 seleccionado: {os.path.basename(p2)}")
    if p1 and p2:
        run_shape_similarity(p1, p2, os.path.basename(p1).split('.')[0], os.path.basename(p2).split('.')[0])
