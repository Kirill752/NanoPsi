import numpy as np
import pyvista as pv
import os
import glob
import re
from scipy.interpolate import RegularGridInterpolator
from visualize_psi import load_psi_data

def align_psi_to_grid(psi, x, y, z, target_x, target_y, target_z):
    if psi.shape == (len(target_x), len(target_y), len(target_z)):
        if np.allclose(x, target_x) and np.allclose(y, target_y) and np.allclose(z, target_z):
            return psi

    interp = RegularGridInterpolator((x, y, z), psi, bounds_error=False, fill_value=0.0)
    
    X, Y, Z = np.meshgrid(target_x, target_y, target_z, indexing='ij')
    points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    
    new_psi_flat = interp(points)
    new_psi = new_psi_flat.reshape(len(target_x), len(target_y), len(target_z))
    
    return new_psi

def get_energy_from_filename(filename):
    match = re.search(r"E_([-+]?\d*\.\d+)", filename)
    if match:
        return float(match.group(1))
    return None

def interpolate_wavefunctions(psi1, psi2, steps=10):
    interpolated_psis = []
    for i in range(steps):
        alpha = i / steps
        psi_interp = (1 - alpha) * psi1 + alpha * psi2
        interpolated_psis.append(psi_interp)
    return interpolated_psis

def animate_wavefunctions(output_filename="wavefunction_animation.gif", fps=10, interpolation_steps=10):
    psi_files = glob.glob("results/psi_*.dat")
    
    if not psi_files:
        print("No wavefunction files found in results/")
        return

    files_with_energy = []
    for f in psi_files:
        e = get_energy_from_filename(f)
        if e is not None:
            files_with_energy.append((e, f))
    
    files_with_energy.sort(key=lambda x: x[0])
    
    if not files_with_energy:
        print("Could not extract energies from filenames.")
        return

    print(f"Found {len(files_with_energy)} files. Sorting by energy:")
    for e, f in files_with_energy:
        print(f"  E = {e:.4f}: {f}")

    first_energy, first_file = files_with_energy[0]
    psi_prev, ref_x, ref_y, ref_z = load_psi_data(first_file)
    
    grid = pv.RectilinearGrid(ref_x, ref_y, ref_z)
    
    plotter = pv.Plotter(off_screen=True)
    plotter.open_gif(output_filename, fps=fps)
    
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

    current_actor = None
    
    print("Starting animation generation with interpolation...")
    
    for i in range(len(files_with_energy) - 1):
        energy1, file1 = files_with_energy[i]
        energy2, file2 = files_with_energy[i+1]
        
        print(f"Interpolating between E={energy1:.4f} and E={energy2:.4f}")
        
        psi1, x1, y1, z1 = load_psi_data(file1)
        psi2, x2, y2, z2 = load_psi_data(file2)
        
        psi1 = align_psi_to_grid(psi1, x1, y1, z1, ref_x, ref_y, ref_z)
        psi2 = align_psi_to_grid(psi2, x2, y2, z2, ref_x, ref_y, ref_z)
        
        psis = interpolate_wavefunctions(psi1, psi2, steps=interpolation_steps)
        
        for j, psi_data in enumerate(psis):
            grid.point_data["probability_density"] = psi_data.flatten(order='F')
            
            max_val = np.max(psi_data)
            levels = [0.1 * max_val, 0.5 * max_val, 0.8 * max_val]
            
            contours = grid.contour(isosurfaces=levels, scalars="probability_density")
            
            if current_actor:
                plotter.remove_actor(current_actor)
                
            current_actor = plotter.add_mesh(contours, opacity=0.5, cmap="viridis", show_scalar_bar=True, label="Probability Density")
            
            alpha = j / interpolation_steps
            current_energy = (1 - alpha) * energy1 + alpha * energy2
            
            plotter.add_title(f"Energy: {current_energy:.4f} eV")
            
            plotter.write_frame()

    last_energy, last_file = files_with_energy[-1]
    psi_last, lx, ly, lz = load_psi_data(last_file)
    psi_last = align_psi_to_grid(psi_last, lx, ly, lz, ref_x, ref_y, ref_z)
    
    grid.point_data["probability_density"] = psi_last.flatten(order='F')
    max_val = np.max(psi_last)
    levels = [0.1 * max_val, 0.5 * max_val, 0.8 * max_val]
    contours = grid.contour(isosurfaces=levels, scalars="probability_density")
    if current_actor:
        plotter.remove_actor(current_actor)
    current_actor = plotter.add_mesh(contours, opacity=0.5, cmap="viridis", show_scalar_bar=True, label="Probability Density")
    plotter.add_title(f"Energy: {last_energy:.4f} eV")
    
    for _ in range(fps):
        plotter.write_frame()

    plotter.close()
    print(f"Animation saved to {output_filename}")

if __name__ == "__main__":
    animate_wavefunctions(fps=10, interpolation_steps=20)