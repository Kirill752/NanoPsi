class PsiDataExporter:
    """Класс для экспорта данных в формате, совместимом с psireader и plots"""
    
    def __init__(self, field_solver, nano_system):
        self.field_solver = field_solver
        self.nano_system = nano_system
    
    def export_to_psi_format(self, filename="nanobridge_potential.dat"):
        """Экспортирует данные потенциала в формате, читаемом psireader"""
        if self.field_solver.potential is None:
            print("Сначала необходимо рассчитать потенциал!")
            return False
        
        X, Y, Z = self.field_solver.grid
        potential = self.field_solver.potential
        
        print(f"Экспорт данных в файл: {filename}")
        print(f"Размер сетки: {X.shape}")
        
        with open(filename, 'w') as f:
            # Записываем данные в формате: x y z U
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    for k in range(X.shape[2]):
                        x, y, z = X[i,j,k], Y[i,j,k], Z[i,j,k]
                        U = potential[i,j,k]
                        f.write(f"{x:.6f} {y:.6f} {z:.6f} {U:.6f}\n")
        
        print(f"Данные успешно экспортированы в {filename}")
        return True
    
    def create_spectra_file(self, filename="spectra_nanobridge.dat", energy_levels=10):
        """Создает файл спектров (заглушка для совместимости)"""
        with open(filename, 'w') as f:
            # Создаем фиктивные уровни энергии для совместимости
            for i in range(energy_levels):
                energy = -0.1 + i * 0.02  # Фиктивные значения
                f.write(f"{energy:.6f}\n")
        print(f"Файл спектров создан: {filename}")
