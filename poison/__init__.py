from poison.view.view import visualize_nanobridge_potential
from poison.view.utils.psi_export import PsiDataExporter
from poison.view.field_view import NanoSystemVisualizer
from poison.lib.model import CompleteNanoSystem
from poison.solver import ElectricFieldSolver
from conf.config import ConfigManager, CONFIGS
import glob
import os


def main():
    # Visualize all single_qd_voltage_sweep configurations
    # config_dir = "conf/voltage_sweep/"
    config_files = ["conf/voltage_sweep/config_double_qd_barrier_0.005V.yaml"]
    
    print(f"Found {len(config_files)} config files to visualize")
    
    for config_file in config_files:
        print(f"\nProcessing: {config_file}")
        config = ConfigManager.load_config(config_file)
        nano_system = CompleteNanoSystem(config)
        nano_system.create_complete_system()
        NanoSystemVisualizer(nano_system).visualize_complete_system()
        print(f"  Visualization saved to: {config['file_path']['visualization_output']}")
    
    print(f"\nCompleted visualization of {len(config_files)} structures")


if __name__ == "__main__":
    main()