from poison.lib.model import CompleteNanoSystem
import numpy as np
import time
import pickle
from numba import jit

@jit(nopython=True)
def _sor_iteration_numba(potential, mask, eps_map, omega, tolerance, max_iter):
    """Оптимизированный SOR решатель с использованием Numba JIT компиляции"""
    nx, ny, nz = potential.shape
    
    for iteration in range(max_iter):
        max_change = 0.0
        
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                for k in range(1, nz-1):
                    if mask[i,j,k]:
                        continue
                    
                    # Получаем диэлектрические проницаемости из предвычисленной карты
                    eps_center = eps_map[i,j,k]
                    eps_x_prev = eps_map[i-1,j,k]
                    eps_x_next = eps_map[i+1,j,k]
                    eps_y_prev = eps_map[i,j-1,k]
                    eps_y_next = eps_map[i,j+1,k]
                    eps_z_prev = eps_map[i,j,k-1]
                    eps_z_next = eps_map[i,j,k+1]
                    
                    # Усредненные значения
                    eps_x_avg_prev = (eps_center + eps_x_prev) * 0.5
                    eps_x_avg_next = (eps_center + eps_x_next) * 0.5
                    eps_y_avg_prev = (eps_center + eps_y_prev) * 0.5
                    eps_y_avg_next = (eps_center + eps_y_next) * 0.5
                    eps_z_avg_prev = (eps_center + eps_z_prev) * 0.5
                    eps_z_avg_next = (eps_center + eps_z_next) * 0.5
                    
                    numerator = (eps_x_avg_prev * potential[i-1,j,k] +
                               eps_x_avg_next * potential[i+1,j,k] +
                               eps_y_avg_prev * potential[i,j-1,k] +
                               eps_y_avg_next * potential[i,j+1,k] +
                               eps_z_avg_prev * potential[i,j,k-1] +
                               eps_z_avg_next * potential[i,j,k+1])
                    
                    denominator = (eps_x_avg_prev + eps_x_avg_next +
                                 eps_y_avg_prev + eps_y_avg_next +
                                 eps_z_avg_prev + eps_z_avg_next)
                    
                    if denominator > 0:
                        new_value = numerator / denominator
                        change = omega * (new_value - potential[i,j,k])
                        potential[i,j,k] += change
                        if abs(change) > max_change:
                            max_change = abs(change)
        
        if max_change < tolerance:
            return potential, iteration + 1, max_change
    
    return potential, max_iter, max_change


class ElectricFieldSolver:
    """Класс для решения уравнения Лапласа и расчета электрического поля"""
    
    def __init__(self, nano_system: CompleteNanoSystem, grid_resolution=50):
        self.nano_system = nano_system
        self.grid_resolution = grid_resolution
        self.potential = None
        self.electric_field = None
        self.grid = None
        self.mask = None
        self.dielectric_constants = {
            'air': 1.0, 'oxide': 3.9, 'nano_bridge': 11.7, 
            'substrate': 11.7, 'gate': 1e10
        }
    
    def create_computational_grid(self):
        """Создает вычислительную сетку"""
        air_box = self.nano_system.air_environment.box
        air_bounds = air_box.GetBounds()
        
        x = np.linspace(air_bounds[0], air_bounds[1], self.grid_resolution)
        y = np.linspace(air_bounds[2], air_bounds[3], self.grid_resolution)
        z = np.linspace(air_bounds[4], air_bounds[5], self.grid_resolution)
        
        self.grid = np.meshgrid(x, y, z, indexing='ij')
        return self.grid
    
    def get_dielectric_constant_at_point(self, point):
        """Возвращает диэлектрическую проницаемость в точке"""
        x, y, z = point
        components = self.nano_system.all_components
        
        # Проверяем материалы в порядке приоритета
        for material_type in ['gate', 'substrate', 'nano_bridge', 'oxide']:
            if material_type in components:
                for box in components[material_type]:
                    bounds = box.GetBounds()
                    x_min, x_max = min(bounds[0], bounds[1]), max(bounds[0], bounds[1])
                    y_min, y_max = min(bounds[2], bounds[3]), max(bounds[2], bounds[3])
                    z_min, z_max = min(bounds[4], bounds[5]), max(bounds[4], bounds[5])
                    
                    if (x_min <= x <= x_max and y_min <= y <= y_max and z_min <= z <= z_max):
                        return self.dielectric_constants[material_type]
        
        return self.dielectric_constants['air']
    
    def create_boundary_mask(self, gate_potential=10.0):
        """Создает маску граничных условий"""
        X, Y, Z = self.grid
        self.mask = np.zeros(X.shape, dtype=bool)
        boundary_values = np.zeros(X.shape)
        
        air_box = self.nano_system.air_environment.box
        air_bounds = air_box.GetBounds()
        tolerance = 1e-6
        
        # Границы воздуха - земля (0V)
        boundary_points = 0
        gate_points = 0
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                for k in range(X.shape[2]):
                    x, y, z = X[i,j,k], Y[i,j,k], Z[i,j,k]
                    
                    # Границы воздуха
                    on_boundary = (
                        abs(x - air_bounds[0]) < tolerance or 
                        abs(x - air_bounds[1]) < tolerance or
                        abs(y - air_bounds[2]) < tolerance or 
                        abs(y - air_bounds[3]) < tolerance or
                        abs(z - air_bounds[4]) < tolerance or 
                        abs(z - air_bounds[5]) < tolerance
                    )
                    
                    if on_boundary:
                        self.mask[i,j,k] = True
                        boundary_values[i,j,k] = 0.0
                        boundary_points += 1
                    
                    # Электрод - заданный потенциал
                    elif self.is_point_in_material((x,y,z), 'gate'):
                        self.mask[i,j,k] = True
                        boundary_values[i,j,k] = gate_potential
                        gate_points += 1
        
        print(f"Точек на электроде: {gate_points:,}")
        print(f"Точек на границах воздуха: {boundary_points:,}")
        return boundary_values
    
    def is_point_in_material(self, point, material_type):
        """Проверяет, находится ли точка внутри указанного материала"""
        x, y, z = point
        components = self.nano_system.all_components
        
        if material_type in components:
            for box in components[material_type]:
                bounds = box.GetBounds()
                x_min, x_max = min(bounds[0], bounds[1]), max(bounds[0], bounds[1])
                y_min, y_max = min(bounds[2], bounds[3]), max(bounds[2], bounds[3])
                z_min, z_max = min(bounds[4], bounds[5]), max(bounds[4], bounds[5])
                
                if (x_min <= x <= x_max and y_min <= y <= y_max and z_min <= z <= z_max):
                    return True
        return False
    
    def precompute_dielectric_map(self):
        """Предвычисляет карту диэлектрических проницаемостей для всей сетки"""
        X, Y, Z = self.grid
        eps_map = np.ones(X.shape, dtype=np.float64) * self.dielectric_constants['air']
        
        print("Предвычисление карты диэлектрических проницаемостей...")
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                for k in range(X.shape[2]):
                    eps_map[i,j,k] = self.get_dielectric_constant_at_point(
                        (X[i,j,k], Y[i,j,k], Z[i,j,k])
                    )
        
        return eps_map
    
    def solve_laplace_sor(self, gate_potential=10.0, omega=1.8, max_iter=1000, tolerance=5e-1, out="electric_field_results.pkl"):
        """Решает уравнение Лапласа методом SOR"""
        print("Создание вычислительной сетки...")
        self.create_computational_grid()
        X, Y, Z = self.grid
        
        print(f"Размер сетки: {X.shape}")
        boundary_values = self.create_boundary_mask(gate_potential)
        self.potential = boundary_values.copy()
        
        # Предвычисляем карту диэлектрических проницаемостей
        eps_map = self.precompute_dielectric_map()
        
        print("Решение уравнения Лапласа...")
        start_time = time.time()
        
        # Используем оптимизированную функцию с Numba
        self.potential, iterations, final_change = _sor_iteration_numba(
            self.potential, self.mask, eps_map, omega, tolerance, max_iter
        )
        
        print(f"Сходимость достигнута за {iterations} итераций, финальное изменение: {final_change:.2e}")
        
        end_time = time.time()
        print(f"Расчет завершен за {end_time - start_time:.2f} секунд")
        
        self.calculate_electric_field()
        self.save_results(out)
        return self.potential
    
    def calculate_electric_field(self):
        """Вычисляет электрическое поле"""
        X, Y, Z = self.grid
        Ex = -np.gradient(self.potential, axis=0)
        Ey = -np.gradient(self.potential, axis=1)
        Ez = -np.gradient(self.potential, axis=2)
        self.electric_field = (Ex, Ey, Ez)
        return self.electric_field
    
    def save_results(self, filename="electric_field_results.pkl"):
        """Сохраняет результаты расчета"""
        if self.potential is None:
            return
            
        data_to_save = {
            'potential': self.potential,
            'electric_field': self.electric_field,
            'grid': self.grid,
            'grid_resolution': self.grid_resolution,
            'mask': self.mask,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(data_to_save, f)
        
        print(f"Результаты сохранены в {filename}")
    
    def load_results(self, filename):
        """Загружает результаты расчета"""
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            
            self.potential = data['potential']
            self.electric_field = data['electric_field']
            self.grid = data['grid']
            self.grid_resolution = data.get('grid_resolution', self.grid_resolution)
            
            print(f"Результаты загружены из {filename}")
            return True
            
        except FileNotFoundError:
            print(f"Файл {filename} не найден!")
            return False