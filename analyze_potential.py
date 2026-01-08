import numpy as np
import matplotlib.pyplot as plt
import os

def load_potential(filename):
    print(f"Loading potential from {filename}...")
    data = np.loadtxt(filename)
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
    
    print(f"Grid: {nx}x{ny}x{nz}")
    
    # Reshape
    V = np.zeros((nx, ny, nz))
    x_map = {v: i for i, v in enumerate(unique_x)}
    y_map = {v: i for i, v in enumerate(unique_y)}
    z_map = {v: i for i, v in enumerate(unique_z)}
    
    for i in range(len(U)):
        xi = x_map[x[i]]
        yi = y_map[y[i]]
        zi = z_map[z[i]]
        V[xi, yi, zi] = U[i]
        
    return V, unique_x, unique_y, unique_z

def analyze():
    filename = "diplom/data/nanobridge_potential.dat"
    try:
        V, x, y, z = load_potential(filename)
    except FileNotFoundError:
        print("File not found")
        return

    min_val = np.min(V)
    max_val = np.max(V)
    min_loc_idx = np.unravel_index(np.argmin(V), V.shape)
    min_loc = (x[min_loc_idx[0]], y[min_loc_idx[1]], z[min_loc_idx[2]])
    
    print(f"Potential range: [{min_val:.4f}, {max_val:.4f}]")
    print(f"Global minimum at: {min_loc} with value {min_val:.4f}")
    
    # Check boundaries
    print("\nBoundary values:")
    print(f"X min face mean: {np.mean(V[0,:,:]):.4f}")
    print(f"X max face mean: {np.mean(V[-1,:,:]):.4f}")
    print(f"Y min face mean: {np.mean(V[:,0,:]):.4f}")
    print(f"Y max face mean: {np.mean(V[:,-1,:]):.4f}")
    print(f"Z min face mean: {np.mean(V[:,:,0]):.4f}")
    print(f"Z max face mean: {np.mean(V[:,:,-1]):.4f}")
    
    # Plot slices through minimum
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(V[:, :, min_loc_idx[2]].T, origin='lower', extent=[x[0], x[-1], y[0], y[-1]], cmap='jet')
    plt.colorbar(label='Potential (eV)')
    plt.title(f'XY Slice at Z={min_loc[2]:.2f}')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    plt.subplot(132)
    plt.imshow(V[:, min_loc_idx[1], :].T, origin='lower', extent=[x[0], x[-1], z[0], z[-1]], cmap='jet')
    plt.colorbar(label='Potential (eV)')
    plt.title(f'XZ Slice at Y={min_loc[1]:.2f}')
    plt.xlabel('X')
    plt.ylabel('Z')
    
    plt.subplot(133)
    plt.plot(x, V[:, min_loc_idx[1], min_loc_idx[2]], label='X-cut')
    plt.plot(y, V[min_loc_idx[0], :, min_loc_idx[2]], label='Y-cut')
    plt.plot(z, V[min_loc_idx[0], min_loc_idx[1], :], label='Z-cut')
    plt.legend()
    plt.title('Cuts through minimum')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/potential_analysis.png')
    print("Saved analysis plot to results/potential_analysis.png")

if __name__ == "__main__":
    analyze()