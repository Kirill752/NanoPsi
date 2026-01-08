import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
import os
import yaml

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

def crop_potential(V, x, y, z, config):
    """
    Crops the potential to the nanobridge region defined in config.
    """
    if config is None:
        print("No config provided, skipping cropping.")
        return V, x, y, z

    print("Cropping potential to nanobridge region...")
    
    nb_config = config['nanobridge']
    
    # Calculate bounds based on nanobridge geometry
    # Center is at (0,0,0) based on model.py
    
    grip_length = nb_config['grip_length']
    grip_width = nb_config['grip_width']
    grip_height = nb_config['grip_height']
    end_length = nb_config['end_length']
    end_width = nb_config['end_width']
    
    # Total length = grip_length + 2 * end_length
    # X range: [-total_length/2, total_length/2]
    total_length = grip_length + 2 * end_length
    x_min = -total_length / 2
    x_max = total_length / 2
    
    # Y range: max width is end_width
    y_min = -end_width / 2
    y_max = end_width / 2
    
    # Z range: [0, grip_height]
    z_min = 0
    z_max = grip_height
    
    # Add a small padding to ensure we capture the boundaries
    padding = 2.0
    x_min -= padding
    x_max += padding
    y_min -= padding
    y_max += padding
    z_min -= padding
    z_max += padding
    
    print(f"Crop bounds: X[{x_min:.2f}, {x_max:.2f}], Y[{y_min:.2f}, {y_max:.2f}], Z[{z_min:.2f}, {z_max:.2f}]")
    
    # Find indices
    x_indices = np.where((x >= x_min) & (x <= x_max))[0]
    y_indices = np.where((y >= y_min) & (y <= y_max))[0]
    z_indices = np.where((z >= z_min) & (z <= z_max))[0]
    
    if len(x_indices) == 0 or len(y_indices) == 0 or len(z_indices) == 0:
        print("Warning: Cropping resulted in empty region! Check bounds.")
        return V, x, y, z
        
    # Slice
    V_cropped = V[x_indices[0]:x_indices[-1]+1,
                  y_indices[0]:y_indices[-1]+1,
                  z_indices[0]:z_indices[-1]+1]
                  
    x_cropped = x[x_indices]
    y_cropped = y[y_indices]
    z_cropped = z[z_indices]
    
    print(f"Cropped grid dimensions: {V_cropped.shape}")
    
    return V_cropped, x_cropped, y_cropped, z_cropped

def load_potential(filename, config=None):
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
        # Fallback: create grid and map values (slower but safer if data is not perfectly ordered)
        V = np.zeros((nx, ny, nz))
        
        # Create mapping from coordinate to index
        x_map = {val: i for i, val in enumerate(unique_x)}
        y_map = {val: i for i, val in enumerate(unique_y)}
        z_map = {val: i for i, val in enumerate(unique_z)}
        
        for i in range(len(U)):
            xi = x_map[x[i]]
            yi = y_map[y[i]]
            zi = z_map[z[i]]
            V[xi, yi, zi] = U[i]

    # Convert electrostatic potential (Volts) to potential energy for electron (eV)
    # U = -e * phi
    # Since we work in eV, the numerical value is just -phi
    # print("Converting electrostatic potential to electron potential energy (U = -V)...")
    # V = -1.0 * V
    
    # Crop if config is provided
    if config:
        V, unique_x, unique_y, unique_z = crop_potential(V, unique_x, unique_y, unique_z, config)
    
    return V, unique_x, unique_y, unique_z

def build_hamiltonian(V, x, y, z, m_eff=0.067):
    """
    Builds the Hamiltonian matrix for the Schrödinger equation.
    H = -hbar^2 / (2*m) * laplacian + V
    
    m_eff: effective mass (in units of electron mass m0)
    """
    print("Building Hamiltonian...")
    
    nx, ny, nz = V.shape
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]
    
    print(f"Grid steps: dx={dx:.4f}, dy={dy:.4f}, dz={dz:.4f}")
    
    # Physical constants
    hbar = 1.0545718e-34 # J*s
    m0 = 9.10938356e-31 # kg
    e = 1.60217663e-19 # C
    
    # Convert units if necessary. 
    # Assuming coordinates are in nm and potential in Volts (eV).
    # We want to work in eV for energy and nm for length.
    # hbar^2 / (2 * m) in units of eV * nm^2
    
    # hbar * c approx 197.3 eV * nm
    # m0 * c^2 = 0.511 MeV = 511000 eV
    # hbar^2 / (2 * m_eff * m0) = (hbar*c)^2 / (2 * m_eff * m0 * c^2)
    
    hbar_c = 197.3269804 # eV * nm
    m0_c2 = 0.510998950e6 # eV
    
    coeff = (hbar_c**2) / (2 * m_eff * m0_c2)
    print(f"Kinetic energy coefficient: {coeff:.6f} eV*nm^2")
    
    N = nx * ny * nz
    
    # Construct Laplacian using finite differences
    # 1D Laplacians
    
    # Diagonals for x
    main_diag_x = -2.0 / dx**2 * np.ones(N)
    off_diag_x = 1.0 / dx**2 * np.ones(N-1)
    # Fix boundaries for x (remove connections between z-layers/y-rows when flattened)
    # Actually, constructing via Kronecker products is cleaner, but let's do sparse diags directly for 3D
    
    # Let's use a simpler approach: construct diagonals for the 3D matrix
    
    # The flattened index idx = i*ny*nz + j*nz + k
    # where i is x-index, j is y-index, k is z-index
    
    # Stride for x neighbors: ny*nz
    # Stride for y neighbors: nz
    # Stride for z neighbors: 1
    
    stride_x = ny * nz
    stride_y = nz
    stride_z = 1
    
    # Potential energy on diagonal
    diagonals = [V.flatten()]
    offsets = [0]
    
    # Kinetic energy parts
    # Center diagonal contribution from kinetic energy: 2/dx^2 + 2/dy^2 + 2/dz^2
    kinetic_diag = coeff * (2/dx**2 + 2/dy**2 + 2/dz**2) * np.ones(N)
    diagonals[0] += kinetic_diag
    
    # Off-diagonals
    # Z-neighbors
    off_z = -coeff / dz**2 * np.ones(N - stride_z)
    # Remove connections at boundaries (every nz-th element)
    # For z, boundaries are at k=0 and k=nz-1.
    # In the flattened array, we just need to be careful not to wrap around?
    # Actually, standard diag construction connects i to i+1.
    # If i is at boundary z_max (k=nz-1), i+1 is (k=0) of next y. We don't want that connection.
    # So we set the coupling to 0 where k=nz-1
    for i in range(nz - 1, N - 1, nz):
        off_z[i] = 0
        
    diagonals.append(off_z)
    offsets.append(stride_z)
    diagonals.append(off_z)
    offsets.append(-stride_z)
    
    # Y-neighbors
    off_y = -coeff / dy**2 * np.ones(N - stride_y)
    # Remove connections at boundaries
    # Boundary is when j=ny-1.
    # Index i corresponds to (ix, iy, iz).
    # We want to cut connection if iy = ny-1.
    # iy = (i // nz) % ny
    for i in range(N - stride_y):
        if ((i // nz) % ny) == ny - 1:
            off_y[i] = 0
            
    diagonals.append(off_y)
    offsets.append(stride_y)
    diagonals.append(off_y)
    offsets.append(-stride_y)
    
    # X-neighbors
    off_x = -coeff / dx**2 * np.ones(N - stride_x)
    # Boundary is when ix = nx-1.
    # ix = i // (ny*nz)
    # Actually for the last block we just stop, so standard diag is fine, 
    # except we don't need to cut internal boundaries because x is the slowest index.
    # Wait, if we have [x0y0z0, ..., x0y0zN, x0y1z0...]
    # x-neighbor of index i is i + stride_x.
    # This is always valid unless i + stride_x >= N.
    # So no special boundary cutting needed inside the array for the slowest dimension.
    
    diagonals.append(off_x)
    offsets.append(stride_x)
    diagonals.append(off_x)
    offsets.append(-stride_x)
    
    H = diags(diagonals, offsets, shape=(N, N), format='csr')
    return H

def solve_schrodinger(H, k=10, sigma=None):
    """
    Solves the eigenvalue problem H*psi = E*psi using Arnoldi method (via ARPACK).
    Returns k smallest eigenvalues and eigenvectors.
    """
    print(f"Solving for {k} lowest eigenvalues...")
    
    # Use 'SA' (Smallest Algebraic) to find the most negative eigenvalues (lowest energy states)
    # If sigma is provided, use shift-invert mode to find eigenvalues near sigma
    if sigma is not None:
        print(f"Using shift-invert mode with sigma={sigma}")
        vals, vecs = eigsh(H, k=k, sigma=sigma, which='LM') # LM of inverted operator => closest to sigma
    else:
        print("Using standard mode (which='SA')")
        vals, vecs = eigsh(H, k=k, which='SA')
    
    return vals, vecs

def save_wavefunctions(vecs, vals, x, y, z, output_dir="results"):
    """
    Saves wavefunctions to files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    nx, ny, nz = len(x), len(y), len(z)
    
    for i in range(len(vals)):
        E = vals[i]
        psi = vecs[:, i].reshape((nx, ny, nz))
        prob_density = np.abs(psi)**2
        
        # Normalize
        norm = np.sum(prob_density) * (x[1]-x[0]) * (y[1]-y[0]) * (z[1]-z[0])
        prob_density /= norm
        
        filename = os.path.join(output_dir, f"psi_{i}_E_{E:.4f}.dat")
        print(f"Saving wavefunction {i} (E={E:.4f} eV) to {filename}...")
        
        # Save in same format as potential: x y z val
        with open(filename, 'w') as f:
            for ix in range(nx):
                for iy in range(ny):
                    for iz in range(nz):
                        f.write(f"{x[ix]:.6f} {y[iy]:.6f} {z[iz]:.6f} {prob_density[ix, iy, iz]:.6e}\n")



def main():
    potential_file = "diplom/data/nanobridge_potential.dat"
    
    # Load config
    config = load_config()
    
    # 1. Load Potential
    try:
        V, x, y, z = load_potential(potential_file, config)
    except FileNotFoundError as e:
        print(e)
        return

    # 2. Build Hamiltonian
    # Effective mass for GaAs is approx 0.067 m0
    H = build_hamiltonian(V, x, y, z, m_eff=0.067)
    
    # 3. Solve
    # Estimate ground state energy near the minimum potential
    min_potential = np.min(V)
    print(f"Minimum potential energy: {min_potential:.4f} eV")
    
    num_levels = 5
    # Use sigma slightly below min potential to ensure we find the ground state
    vals, vecs = solve_schrodinger(H, k=num_levels, sigma=min_potential - 0.1)
    
    print("\nCalculated Energy Levels (eV):")
    for i, E in enumerate(vals):
        print(f"Level {i}: {E:.6f} eV")
        
    # 4. Save Results
    save_wavefunctions(vecs, vals, x, y, z)
    
    # Optional: Plot spectrum
    plt.figure(figsize=(8, 6))
    plt.plot(range(num_levels), vals, 'bo-', label='Energy Levels')
    plt.xlabel('Level Index')
    plt.ylabel('Energy (eV)')
    plt.title('Energy Spectrum of Electron in Nanobridge Potential')
    plt.grid(True)
    plt.savefig('results/energy_spectrum.png')
    print("Spectrum plot saved to results/energy_spectrum.png")

if __name__ == "__main__":
    main()
