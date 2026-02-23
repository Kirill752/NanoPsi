from poison.view.view import visualize_nanobridge_potential
from poison.view.utils.psi_export import PsiDataExporter
from poison.view.field_view import NanoSystemVisualizer
from poison.lib.model import CompleteNanoSystem
from poison.solver import ElectricFieldSolver
from conf.config import ConfigManager

def main():
    config = ConfigManager.load_config("conf/config.yaml")
    nano_system = CompleteNanoSystem(config)
    nano_system.create_complete_system()

    NanoSystemVisualizer(nano_system).visualize_complete_system()
    
    field_solver = ElectricFieldSolver(nano_system, grid_resolution=config["solver"]["grid_resolution"])
    field_solver.solve_laplace_sor(
        gate_potential=config['electrode']["potential"],
        omega=config["solver"]["relaxation_parameter"],
        max_iter=config["solver"]["max_iterations"],
        tolerance=config["solver"]["tolerance"],
        out=config["file_path"]["results_file"]
    )
    
    visualize_nanobridge_potential(config, nano_system, save_plots=True)


    exporter = PsiDataExporter(field_solver, nano_system)
    exporter.export_to_psi_format(config["file_path"]["potential_data_file"])
    exporter.create_spectra_file(config["file_path"]["spectra_file"])
    # print("Загрузка конфигурации...")
    # config = ConfigManager.load_config("conf/config.yaml")
    
    # print("Создание модели наносистемы...")
    # nano_system = CompleteNanoSystem(config)
    # nano_system.create_complete_system()
    
    # print("\nВизуализация профилей потенциала...")
    # visualize_nanobridge_potential(config, nano_system, save_plots=True)
    
    # print("\nГотово! Графики сохранены в results/potential_profiles.png")

if __name__ == "__main__":
    main()