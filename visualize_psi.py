import numpy as np
import pyvista as pv
import os
import glob

from poison.view.field_view import NanoSystemVisualizer
from poison.lib.model import CompleteNanoSystem
from conf.config import ConfigManager

def load_psi_data(filename):
    print(f"Loading wavefunction from {filename}...")
    data = np.loadtxt(filename)
    
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    val = data[:, 3]
    
    unique_x = np.unique(x)
    unique_y = np.unique(y)
    unique_z = np.unique(z)
    
    nx = len(unique_x)
    ny = len(unique_y)
    nz = len(unique_z)
    
    print(f"Grid dimensions: {nx}x{ny}x{nz}")
    
    grid_val = np.zeros((nx, ny, nz))
    
    x_map = {v: i for i, v in enumerate(unique_x)}
    y_map = {v: i for i, v in enumerate(unique_y)}
    z_map = {v: i for i, v in enumerate(unique_z)}
    
    for i in range(len(val)):
        xi = x_map[x[i]]
        yi = y_map[y[i]]
        zi = z_map[z[i]]
        grid_val[xi, yi, zi] = val[i]
        
    return grid_val, unique_x, unique_y, unique_z

def visualize_wavefunction(filename, isosurface_value=0.1):
    try:
        psi, x, y, z = load_psi_data(filename)
    except FileNotFoundError:
        print(f"File not found: {filename}")
        return

    grid = pv.RectilinearGrid(x, y, z)
    
    grid.point_data["probability_density"] = psi.flatten(order='F')
    
    plotter = pv.Plotter()
    
    # Add nanostructure visualization
    try:
        config = ConfigManager.load_config("conf/config.yaml")
        nano_system = CompleteNanoSystem(config)
        nano_system.create_complete_system()
        
        visualizer = NanoSystemVisualizer(nano_system)
        pyvista_objects = visualizer.convert_to_pyvista()
        
        added_to_legend = {
            'nano_bridge': False,
            'oxide': False,
            'substrate': False,
            'air': False,
            'gate': False
        }
        
        for obj in pyvista_objects:
            # Рисуем только границы структуры (wireframe), чтобы не перекрывать волновую функцию
            if obj['type'] == 'air':
                # Воздух не рисуем вообще
                continue
            
            if obj['type'] in added_to_legend and not added_to_legend[obj['type']]:
                label = obj['name']
                added_to_legend[obj['type']] = True
            else:
                label = None
            
            # Рисуем только рёбра (wireframe) без заполнения
            plotter.add_mesh(obj['mesh'],
                           color=obj['color'],
                           style='wireframe',  # Только границы
                           line_width=2,
                           opacity=1.0,
                           label=label)
                           
    except Exception as e:
        print(f"Warning: Could not load nanostructure visualization: {e}")

    max_val = np.max(psi)
    print(f"Max probability density: {max_val:.6e}")
    
    levels = [0.1 * max_val, 0.5 * max_val, 0.8 * max_val]
    contours = grid.contour(isosurfaces=levels, scalars="probability_density")
    
    plotter.add_mesh(contours, opacity=0.8, cmap="viridis", show_scalar_bar=True, label="Probability Density")
    
    plotter.add_mesh(grid.outline(), color="k")
    
    plotter.show_axes()
    plotter.show_grid(
        xtitle="X (nm)",
        ytitle="Y (nm)",
        ztitle="Z (nm)",
        color='black',
        font_size=10,
        show_xaxis=True,
        show_yaxis=True,
        show_zaxis=True
    )
    
    plotter.add_title(f"Wavefunction: {os.path.basename(filename)}")
    plotter.add_legend()
    
    output_image = filename.replace(".dat", ".png")
    plotter.show(screenshot=output_image, auto_close=False)
    print(f"Saved visualization to {output_image}")
    plotter.close()

def main():
    psi_files = glob.glob("results/psi_*.dat")
    psi_files.sort()
    
    if not psi_files:
        print("No wavefunction files found in results/")
        return
        
    print(f"Found {len(psi_files)} wavefunction files.")
    
    for psi_file in psi_files:
        print(f"Visualizing: {psi_file}")
        visualize_wavefunction(psi_file)

if __name__ == "__main__":
    main()