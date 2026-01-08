from view.view import visualize_nanobridge_potential

from view.utils.psi_export import PsiDataExporter
from view.field_view import NanoSystemVisualizer
from conf.config import ConfigManager
from lib.model import CompleteNanoSystem
from solver import ElectricFieldSolver

def main():
    config = ConfigManager.load_config("conf/config.yaml")
    nano_system = CompleteNanoSystem(config)
    nano_system.create_complete_system()

    NanoSystemVisualizer(nano_system).visualize_complete_system()
    
    field_solver = ElectricFieldSolver(nano_system, grid_resolution=50)
    field_solver.solve_laplace_sor(gate_potential=config['electrode']["potential"], out=config["results_file"])
    
    visualize_nanobridge_potential(config, nano_system, save_plots=True)


    exporter = PsiDataExporter(field_solver, nano_system)
    exporter.export_to_psi_format(config["potential_data_file"])
    exporter.create_spectra_file(config["spectra_file"])

if __name__ == "__main__":
    main()