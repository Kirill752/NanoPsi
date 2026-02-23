import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
import os
import yaml

def load_config(config_path="diplom/conf/config.yaml"):
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        alt_path = "conf/config.yaml"
        if os.path.exists(alt_path):
            with open(alt_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            print(f"Warning: Config file not found at {config_path} or {alt_path}")
            return None

def crop_potential(V, x, y, z, config):
    if config is None:
        print("No config provided, skipping cropping.")
        return V, x, y, z

    print("Cropping potential to nanobridge region...")
    
    nb_config = config['nanobridge']
    
    grip_length = nb_config['grip_length']
    grip_width = nb_config['grip_width']
    grip_height = nb_config['grip_height']
    end_length = nb_config['end_length']
    oxide_thickness = nb_config.get('oxide_thickness', 2.0)
    
    # Обрезаем СТРОГО внутри наномостика (Si), не захватывая электрод!
    # Включаем весь наномостик + резервуары
    total_length = grip_length + 2 * end_length
    x_min = -total_length / 2
    x_max = total_length / 2
    
    # По Y: только внутри Si (без оксида и электрода)
    y_min = -grip_width / 2
    y_max = grip_width / 2
    
    # По Z: от подложки до верха Si (НЕ включая оксид и электрод!)
    z_min = 0  # Граница подложка-Si
    z_max = grip_height  # Верх Si (ниже оксида и электрода)
    
    print(f"Crop bounds: X[{x_min:.2f}, {x_max:.2f}], Y[{y_min:.2f}, {y_max:.2f}], Z[{z_min:.2f}, {z_max:.2f}]")
    
    x_indices = np.where((x >= x_min) & (x <= x_max))[0]
    y_indices = np.where((y >= y_min) & (y <= y_max))[0]
    z_indices = np.where((z >= z_min) & (z <= z_max))[0]
    
    if len(x_indices) == 0 or len(y_indices) == 0 or len(z_indices) == 0:
        print("Warning: Cropping resulted in empty region! Check bounds.")
        return V, x, y, z
        
    V_cropped = V[x_indices[0]:x_indices[-1]+1,
                  y_indices[0]:y_indices[-1]+1,
                  z_indices[0]:z_indices[-1]+1]
                   
    x_cropped = x[x_indices]
    y_cropped = y[y_indices]
    z_cropped = z[z_indices]
    
    print(f"Cropped grid dimensions: {V_cropped.shape}")
    
    return V_cropped, x_cropped, y_cropped, z_cropped

def load_potential(filename, config=None):
    print(f"Loading potential from {filename}...")
    try:
        data = np.loadtxt(filename)
    except OSError:
        alt_filename = os.path.join("diplom", filename)
        if os.path.exists(alt_filename):
             print(f"File not found at {filename}, trying {alt_filename}")
             data = np.loadtxt(alt_filename)
        else:
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

    unique_x = np.unique(x)
    unique_y = np.unique(y)
    unique_z = np.unique(z)

    nx = len(unique_x)
    ny = len(unique_y)
    nz = len(unique_z)

    print(f"Original grid dimensions: nx={nx}, ny={ny}, nz={nz}")

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

    if config:
        V, unique_x, unique_y, unique_z = crop_potential(V, unique_x, unique_y, unique_z, config)
    
    return V, unique_x, unique_y, unique_z

def build_hamiltonian(V, x, y, z, m_eff=0.067):
    print("Building Hamiltonian...")
    
    nx, ny, nz = V.shape
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]
    
    print(f"Grid steps: dx={dx:.4f}, dy={dy:.4f}, dz={dz:.4f}")
    
    hbar_c = 197.3269804
    m0_c2 = 0.510998950e6
    
    coeff = (hbar_c**2) / (2 * m_eff * m0_c2)
    print(f"Kinetic energy coefficient: {coeff:.6f} eV*nm^2")
    
    N = nx * ny * nz
    
    stride_x = ny * nz
    stride_y = nz
    stride_z = 1
    
    # ВАЖНО: V содержит электростатический потенциал φ (в вольтах)
    # Потенциальная энергия электрона: U = q·φ = (-e)·φ
    # В единицах эВ: U [эВ] = (-e)·φ[В] / e = -φ
    # Заряд электрона e=1.602e-19 Кл уже учтен в определении эВ!
    kinetic_diag = coeff * (2/dx**2 + 2/dy**2 + 2/dz**2)
    
    diagonals = [(-V.flatten() + kinetic_diag)]
    offsets = [0]
    
    off_z = -coeff / dz**2 * np.ones(N - stride_z)
    for i in range(nz - 1, N - 1, nz):
        off_z[i] = 0
        
    diagonals.append(off_z)
    offsets.append(stride_z)
    diagonals.append(off_z)
    offsets.append(-stride_z)
    
    off_y = -coeff / dy**2 * np.ones(N - stride_y)
    for i in range(N - stride_y):
        if ((i // nz) % ny) == ny - 1:
            off_y[i] = 0
            
    diagonals.append(off_y)
    offsets.append(stride_y)
    diagonals.append(off_y)
    offsets.append(-stride_y)
    
    off_x = -coeff / dx**2 * np.ones(N - stride_x)
    
    diagonals.append(off_x)
    offsets.append(stride_x)
    diagonals.append(off_x)
    offsets.append(-stride_x)
    
    H = diags(diagonals, offsets, shape=(N, N), format='csr')
    return H

def solve_schrodinger(H, k=10, sigma=None):
    print(f"Solving for {k} lowest eigenvalues...")
    
    if sigma is not None:
        print(f"Using shift-invert mode with sigma={sigma}")
        vals, vecs = eigsh(H, k=k, sigma=sigma, which='LM')
    else:
        print("Using standard mode (which='SA')")
        vals, vecs = eigsh(H, k=k, which='SA')
    
    return vals, vecs

def save_wavefunctions(vecs, vals, x, y, z, output_dir="results"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    nx, ny, nz = len(x), len(y), len(z)
    
    print(f"\nGrid dimensions: nx={nx}, ny={ny}, nz={nz}")
    print(f"Eigenvector shape: {vecs.shape}")
    
    for i in range(len(vals)):
        E = vals[i]
        
        # Reshape с правильным порядком
        psi = vecs[:, i].reshape((nx, ny, nz), order='C')
        prob_density = np.abs(psi)**2
        
        # Нормализация
        dx = x[1] - x[0] if len(x) > 1 else 1.0
        dy = y[1] - y[0] if len(y) > 1 else 1.0
        dz = z[1] - z[0] if len(z) > 1 else 1.0
        
        norm = np.sum(prob_density) * dx * dy * dz
        if norm > 0:
            prob_density /= norm
        
        print(f"Wavefunction {i}: E={E:.6f} eV, norm={norm:.6e}, max(|psi|^2)={np.max(prob_density):.6e}")
        
        filename = os.path.join(output_dir, f"psi_{i}_E_{E:.4f}.dat")
        print(f"Saving to {filename}...")
        
        with open(filename, 'w') as f:
            for ix in range(nx):
                for iy in range(ny):
                    for iz in range(nz):
                        f.write(f"{x[ix]:.6f} {y[iy]:.6f} {z[iz]:.6f} {prob_density[ix, iy, iz]:.6e}\n")



def main():
    potential_file = "nanobridge_potential.dat"
    
    config = load_config()
    
    try:
        V, x, y, z = load_potential(potential_file, config)
    except FileNotFoundError as e:
        print(e)
        return

    H = build_hamiltonian(V, x, y, z, m_eff=0.067)
    
    # V содержит электростатический потенциал φ (в вольтах)
    # Потенциальная энергия электрона: U = -e·φ = -φ (в эВ)
    min_phi = np.min(V)
    max_phi = np.max(V)
    mean_phi = np.mean(V)
    min_U = -max_phi  # Минимум U там, где φ максимален (электрод)
    max_U = -min_phi
    
    print(f"\nЭлектростатический потенциал в обрезанной области:")
    print(f"  φ_min = {min_phi:.4f} V")
    print(f"  φ_max = {max_phi:.4f} V")
    print(f"  φ_mean = {mean_phi:.4f} V")
    print(f"\nПотенциальная энергия электрона (U = -e·φ = -φ в эВ):")
    print(f"  U_min = {min_U:.4f} eV (электрод притягивает)")
    print(f"  U_max = {max_U:.4f} eV")
    print(f"\nОжидаемая энергия основного состояния: чуть выше {min_U:.4f} eV")
    
    num_levels = 5
    vals, vecs = solve_schrodinger(H, k=num_levels, sigma=min_U + 0.01)
    
    print("\nCalculated Energy Levels (eV):")
    for i, E in enumerate(vals):
        print(f"Level {i}: {E:.6f} eV")
        
    save_wavefunctions(vecs, vals, x, y, z)
    
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
