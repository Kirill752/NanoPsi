import numpy as np
import os
import glob

def load_psi_data(filename):
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

def analyze_physics():
    # 1. Load Potential (Cropped)
    # We need to reload the potential to compare locations
    # Since we don't save the cropped potential separately, let's just look at the wavefunction peak
    # and compare it with our knowledge of the geometry.
    
    # Geometry from config (approximate)
    # Center is (0,0,0)
    # Gate is likely above the bridge.
    # Bridge is at Z=0 to Z=6 (grip_height)
    # Gate potential is +10V.
    # Electron (negative charge) should be attracted to positive potential.
    # So electron should be localized where potential is HIGHEST (most positive).
    
    psi_files = glob.glob("results/psi_0_E_*.dat")
    if not psi_files:
        print("No wavefunction files found.")
        return
        
    psi_file = psi_files[0]
    print(f"Analyzing ground state: {psi_file}")
    
    psi_sq, x, y, z = load_psi_data(psi_file)
    
    # Find peak of wavefunction
    max_idx = np.unravel_index(np.argmax(psi_sq), psi_sq.shape)
    peak_loc = (x[max_idx[0]], y[max_idx[1]], z[max_idx[2]])
    
    print(f"Electron probability density peak at: X={peak_loc[0]:.2f}, Y={peak_loc[1]:.2f}, Z={peak_loc[2]:.2f}")
    
    # Check if this makes sense
    print("\nPhysical Consistency Check:")
    print(f"1. X-coordinate ({peak_loc[0]:.2f}): Should be near 0 (center of bridge length).")
    if abs(peak_loc[0]) < 5.0:
        print("   -> OK: Centered along the bridge length.")
    else:
        print("   -> WARNING: Off-center along bridge length.")
        
    print(f"2. Y-coordinate ({peak_loc[1]:.2f}): Should be near 0 (center of bridge width).")
    if abs(peak_loc[1]) < 3.0:
        print("   -> OK: Centered along the bridge width.")
    else:
        print("   -> WARNING: Off-center along bridge width.")
        
    print(f"3. Z-coordinate ({peak_loc[2]:.2f}): Should be near the top surface of the bridge or oxide interface.")
    # Bridge is 0 to 6 nm. Gate is above.
    # Electron should be pulled towards the gate, so it should be at the top of the bridge (near Z=6).
    if 4.0 <= peak_loc[2] <= 8.0:
        print("   -> OK: Localized near the top of the bridge (closest to gate).")
    else:
        print(f"   -> WARNING: Unexpected Z location. Bridge height is ~6nm.")

if __name__ == "__main__":
    analyze_physics()