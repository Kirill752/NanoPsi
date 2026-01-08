import numpy as np
import yaml
import os
import matplotlib.pyplot as plt

def load_config(config_path="diplom/conf/config.yaml"):
    """Loads configuration from yaml file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        # Try alternative path
        alt_path = "conf/config.yaml"
        if os.path.exists(alt_path):
            with open(alt_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            print(f"Warning: Config file not found at {config_path} or {alt_path}")
            return None

def load_potential(filename):
    """
    Loads potential data from a file.
    Format: x y z U
    """
    print(f"Loading potential from {filename}...")
    try:
        data = np.loadtxt(filename)
    except OSError:
        # Try alternative path if file not found
        alt_filename = os.path.join("diplom", filename)
        if os.path.exists(alt_filename):
             print(f"File not found at {filename}, trying {alt_filename}")
             data = np.loadtxt(alt_filename)
        else:
             # Try data directory
             alt_filename_2 = os.path.join("diplom", "data", os.path.basename(filename))
             if os.path.exists(alt_filename_2):
                 print(f"File not found at {filename}, trying {alt_filename_2}")
                 data = np.loadtxt(alt_filename_2)
             else:
                 raise FileNotFoundError(f"Could not find potential file: {filename}")

    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    U = data[:, 3]

    # Determine grid dimensions
    unique_x = np.unique(x)
    unique_y = np.unique(y)
    unique_z = np.unique(z)

    nx = len(unique_x)
    ny = len(unique_y)
    nz = len(unique_z)

    print(f"Original grid dimensions: nx={nx}, ny={ny}, nz={nz}")

    # Reshape potential to 3D array
    try:
        V = U.reshape((nx, ny, nz))
    except ValueError:
        print("Warning: Could not reshape directly. Checking data consistency...")
        V = np.zeros((nx, ny, nz))
        x_map = {val: i for i, val in enumerate(unique_x)}
        y_map = {val: i for i, val in enumerate(unique_y)}
        z_map = {val: i for i, val in enumerate(unique_z)}
        
        for i in range(len(U)):
            xi = x_map[x[i]]
            yi = y_map[y[i]]
            zi = z_map[z[i]]
            V[xi, yi, zi] = U[i]

    return V, unique_x, unique_y, unique_z

def extract_and_analyze():
    config = load_config()
    if not config:
        return

    potential_file = "diplom/data/nanobridge_potential.dat"
    V, x, y, z = load_potential(potential_file)
    
    # Nanobridge geometry
    nb_config = config['nanobridge']
    grip_length = nb_config['grip_length']
    grip_width = nb_config['grip_width']
    grip_height = nb_config['grip_height']
    
    # Define bounds for the grip part (narrow part) of the nanobridge
    # Center is at (0,0,0)
    x_min, x_max = -grip_length/2, grip_length/2
    y_min, y_max = -grip_width/2, grip_width/2
    z_min, z_max = 0, grip_height
    
    print(f"\nExtracting potential inside nanobridge grip:")
    print(f"X: [{x_min}, {x_max}]")
    print(f"Y: [{y_min}, {y_max}]")
    print(f"Z: [{z_min}, {z_max}]")
    
    # Find indices
    x_indices = np.where((x >= x_min) & (x <= x_max))[0]
    y_indices = np.where((y >= y_min) & (y <= y_max))[0]
    z_indices = np.where((z >= z_min) & (z <= z_max))[0]
    
    # Slice
    V_nb = V[x_indices[0]:x_indices[-1]+1, 
             y_indices[0]:y_indices[-1]+1, 
             z_indices[0]:z_indices[-1]+1]
             
    x_nb = x[x_indices]
    y_nb = y[y_indices]
    z_nb = z[z_indices]
    
    print(f"Extracted grid size: {V_nb.shape}")
    
    # Save extracted potential
    output_file = "results/nanobridge_internal_potential.dat"
    if not os.path.exists("results"):
        os.makedirs("results")
        
    with open(output_file, 'w') as f:
        for ix in range(len(x_nb)):
            for iy in range(len(y_nb)):
                for iz in range(len(z_nb)):
                    f.write(f"{x_nb[ix]:.6f} {y_nb[iy]:.6f} {z_nb[iz]:.6f} {V_nb[ix, iy, iz]:.6f}\n")
    print(f"Saved extracted potential to {output_file}")
    
    # Analyze potential distribution
    # We are looking for the potential energy minimum for the electron.
    # Electron potential energy U = -V (if V is electrostatic potential).
    # So we look for the MAXIMUM of V (electrostatic potential).
    
    max_V = np.max(V_nb)
    max_idx = np.unravel_index(np.argmax(V_nb), V_nb.shape)
    max_loc = (x_nb[max_idx[0]], y_nb[max_idx[1]], z_nb[max_idx[2]])
    
    print(f"\nAnalysis of Electrostatic Potential inside Nanobridge:")
    print(f"Maximum Potential (Deepest Well for Electron): {max_V:.4f} V")
    print(f"Location of Maximum: X={max_loc[0]:.2f}, Y={max_loc[1]:.2f}, Z={max_loc[2]:.2f}")
    
    # Check symmetry along Y
    # Take a slice at X=max_loc[0] and Z=max_loc[2]
    y_slice = V_nb[max_idx[0], :, max_idx[2]]
    
    plt.figure(figsize=(10, 6))
    plt.plot(y_nb, y_slice, 'b-o')
    plt.axvline(x=0, color='k', linestyle='--', label='Center (Y=0)')
    plt.axvline(x=max_loc[1], color='r', linestyle='--', label=f'Max (Y={max_loc[1]:.2f})')
    plt.xlabel('Y (nm)')
    plt.ylabel('Electrostatic Potential (V)')
    plt.title(f'Potential Profile across Nanobridge Width (at X={max_loc[0]:.2f}, Z={max_loc[2]:.2f})')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/potential_profile_Y.png')
    print("Saved potential profile plot to results/potential_profile_Y.png")
    
    # Calculate asymmetry
    # Compare potential at Y = +d and Y = -d
    d = 2.0 # nm
    # Find indices closest to +d and -d
    idx_plus = np.abs(y_nb - d).argmin()
    idx_minus = np.abs(y_nb - (-d)).argmin()
    
    val_plus = y_slice[idx_plus]
    val_minus = y_slice[idx_minus]
    
    print(f"\nAsymmetry Check:")
    print(f"Potential at Y ~ {y_nb[idx_plus]:.2f} nm: {val_plus:.4f} V")
    print(f"Potential at Y ~ {y_nb[idx_minus]:.2f} nm: {val_minus:.4f} V")
    print(f"Difference (Right - Left): {val_plus - val_minus:.4f} V")
    
    if val_plus > val_minus:
        print("Conclusion: Potential is higher on the RIGHT side (Y > 0). Electron is pulled to the RIGHT.")
    elif val_minus > val_plus:
        print("Conclusion: Potential is higher on the LEFT side (Y < 0). Electron is pulled to the LEFT.")
    else:
        print("Conclusion: Potential is symmetric.")

if __name__ == "__main__":
    extract_and_analyze()