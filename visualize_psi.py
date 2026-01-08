import numpy as np
import pyvista as pv
import os
import glob

def load_psi_data(filename):
    """
    Loads wavefunction data from a file.
    Format: x y z val
    """
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
    
    # Reshape to 3D grid
    # Assuming the same order as in arnoldi.py (x, y, z)
    # But we need to be careful. Let's use the mapping method to be safe.
    
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
    """
    Visualizes the wavefunction using PyVista.
    """
    try:
        psi, x, y, z = load_psi_data(filename)
    except FileNotFoundError:
        print(f"File not found: {filename}")
        return

    # Create PyVista grid
    grid = pv.RectilinearGrid(x, y, z)
    
    # Add data to grid (flatten in Fortran order because PyVista/VTK uses x-fastest, 
    # but our array is [x,y,z]. Wait, VTK uses x-fastest? 
    # Actually, PyVista wraps VTK. 
    # If we have array[nx, ny, nz], and we pass it to grid.point_data, 
    # we need to flatten it correctly.
    # grid.points are ordered: (x0,y0,z0), (x1,y0,z0), ... (fastest x)
    # So if our array is indexed [xi, yi, zi], we should flatten with order='F' 
    # IF the loops were for k in z: for j in y: for i in x.
    # But our load loop was arbitrary.
    # Let's look at how we filled it: grid_val[xi, yi, zi]
    # If we flatten with order='F', we get elements (0,0,0), (1,0,0), ... which matches x-fastest.
    
    grid.point_data["probability_density"] = psi.flatten(order='F')
    
    # Create plotter
    plotter = pv.Plotter()
    
    # Add isosurfaces
    # Calculate max value to scale isosurfaces
    max_val = np.max(psi)
    print(f"Max probability density: {max_val:.6e}")
    
    # Create contours at different levels
    levels = [0.1 * max_val, 0.5 * max_val, 0.8 * max_val]
    contours = grid.contour(isosurfaces=levels, scalars="probability_density")
    
    plotter.add_mesh(contours, opacity=0.5, cmap="viridis", show_scalar_bar=True, label="Probability Density")
    
    # Add outline
    plotter.add_mesh(grid.outline(), color="k")
    
    # Add axes with labels
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
    
    # Save screenshot
    output_image = filename.replace(".dat", ".png")
    plotter.show(screenshot=output_image, auto_close=False) # auto_close=False to keep window open if interactive
    print(f"Saved visualization to {output_image}")
    plotter.close()

def main():
    # Find all psi files in results directory
    psi_files = glob.glob("results/psi_*.dat")
    psi_files.sort()
    
    if not psi_files:
        print("No wavefunction files found in results/")
        return
        
    print(f"Found {len(psi_files)} wavefunction files.")
    
    # Visualize all found wavefunctions
    for psi_file in psi_files:
        print(f"Visualizing: {psi_file}")
        visualize_wavefunction(psi_file)

if __name__ == "__main__":
    main()