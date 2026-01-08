import numpy as np
import pyvista as pv
import os
import glob
import re
from visualize_psi import load_psi_data

def get_energy_from_filename(filename):
    """
    Extracts energy value from filename.
    Expected format: ..._E_{energy}.dat
    """
    match = re.search(r"E_([-+]?\d*\.\d+)", filename)
    if match:
        return float(match.group(1))
    return None

def interpolate_wavefunctions(psi1, psi2, steps=10):
    """
    Linearly interpolates between two wavefunctions.
    """
    interpolated_psis = []
    for i in range(steps):
        alpha = i / steps
        psi_interp = (1 - alpha) * psi1 + alpha * psi2
        interpolated_psis.append(psi_interp)
    return interpolated_psis

def animate_wavefunctions(output_filename="wavefunction_animation.gif", fps=10, interpolation_steps=10):
    """
    Creates an animation of wavefunctions sorted by energy with interpolation.
    """
    # Find all psi files
    psi_files = glob.glob("results/psi_*.dat")
    
    if not psi_files:
        print("No wavefunction files found in results/")
        return

    # Sort files by energy
    files_with_energy = []
    for f in psi_files:
        e = get_energy_from_filename(f)
        if e is not None:
            files_with_energy.append((e, f))
    
    # Sort by energy (ascending)
    files_with_energy.sort(key=lambda x: x[0])
    
    if not files_with_energy:
        print("Could not extract energies from filenames.")
        return

    print(f"Found {len(files_with_energy)} files. Sorting by energy:")
    for e, f in files_with_energy:
        print(f"  E = {e:.4f}: {f}")

    # Load the first file to setup the grid
    first_energy, first_file = files_with_energy[0]
    psi_prev, x, y, z = load_psi_data(first_file)
    
    # Create PyVista grid
    grid = pv.RectilinearGrid(x, y, z)
    
    # Setup plotter
    plotter = pv.Plotter(off_screen=True)
    plotter.open_gif(output_filename, fps=fps)
    
    # Add outline
    plotter.add_mesh(grid.outline(), color="k")
    
    # Add axes
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

    # We need to keep track of the mesh actor to remove/update it
    current_actor = None
    
    print("Starting animation generation with interpolation...")
    
    # Calculate global max value for consistent isosurfaces
    # This might be expensive if files are large, but good for consistency.
    # Alternatively, we can scale per frame or use the first frame's max.
    # Let's use the first frame's max to keep levels consistent relative to the first state,
    # OR better, find the global max across all files.
    # For now, let's stick to per-frame max or maybe just use the first one.
    # Let's use the max of the current frame to define levels, so we always see something.
    
    for i in range(len(files_with_energy) - 1):
        energy1, file1 = files_with_energy[i]
        energy2, file2 = files_with_energy[i+1]
        
        print(f"Interpolating between E={energy1:.4f} and E={energy2:.4f}")
        
        psi1, _, _, _ = load_psi_data(file1)
        psi2, _, _, _ = load_psi_data(file2)
        
        # Interpolate
        psis = interpolate_wavefunctions(psi1, psi2, steps=interpolation_steps)
        
        for j, psi_data in enumerate(psis):
            # Update grid data
            grid.point_data["probability_density"] = psi_data.flatten(order='F')
            
            # Calculate max value for isosurfaces
            max_val = np.max(psi_data)
            levels = [0.1 * max_val, 0.5 * max_val, 0.8 * max_val]
            
            # Generate contours
            contours = grid.contour(isosurfaces=levels, scalars="probability_density")
            
            # Clear previous mesh
            if current_actor:
                plotter.remove_actor(current_actor)
                
            # Add new mesh
            current_actor = plotter.add_mesh(contours, opacity=0.5, cmap="viridis", show_scalar_bar=True, label="Probability Density")
            
            # Interpolated energy for title
            alpha = j / interpolation_steps
            current_energy = (1 - alpha) * energy1 + alpha * energy2
            
            # Update title
            plotter.add_title(f"Energy: {current_energy:.4f} eV")
            
            # Write frame
            plotter.write_frame()

    # Add the last frame
    last_energy, last_file = files_with_energy[-1]
    psi_last, _, _, _ = load_psi_data(last_file)
    grid.point_data["probability_density"] = psi_last.flatten(order='F')
    max_val = np.max(psi_last)
    levels = [0.1 * max_val, 0.5 * max_val, 0.8 * max_val]
    contours = grid.contour(isosurfaces=levels, scalars="probability_density")
    if current_actor:
        plotter.remove_actor(current_actor)
    current_actor = plotter.add_mesh(contours, opacity=0.5, cmap="viridis", show_scalar_bar=True, label="Probability Density")
    plotter.add_title(f"Energy: {last_energy:.4f} eV")
    
    # Hold the last frame for a bit
    for _ in range(fps):
        plotter.write_frame()

    plotter.close()
    print(f"Animation saved to {output_filename}")

if __name__ == "__main__":
    animate_wavefunctions(fps=10, interpolation_steps=20)