import pickle
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import os

class NanobridgePotentialVisualizer:
    """Специализированный визуализатор потенциала внутри наномостика под электродом"""
    
    def __init__(self, config, nano_system):

        self.config = config
        self.nano_system = nano_system

        file_paths = self.config['file_paths']
        self.pkl_file = file_paths['results_file']

        self.data = None
        self.grid = None
        self.potential = None
        self.nanobridge_mask = None
        self.electrode_mask = None

        output_dir = file_paths.get('output_directory', 'results')
        os.makedirs(output_dir, exist_ok=True)
    
    def load_data(self):
        """Загружает данные из pkl файла"""
        if not os.path.exists(self.pkl_file):
            print(f"Файл {self.pkl_file} не найден!")
            return False
            
        try:
            with open(self.pkl_file, 'rb') as f:
                self.data = pickle.load(f)
            
            self.potential = self.data['potential']
            self.grid = self.data['grid']
            
            print(f"Данные успешно загружены из {self.pkl_file}")
            print(f"Размер сетки: {self.potential.shape}")
            print(f"Диапазон потенциалов: {self.potential.min():.3f} - {self.potential.max():.3f} V")
            
            return True
            
        except Exception as e:
            print(f"Ошибка при загрузке файла: {e}")
            return False
    
    def create_nanobridge_mask(self):
        """Создает маску для области наномостика под электродом используя параметры из конфига"""
        if self.grid is None:
            print("Сначала загрузите данные!")
            return False
            
        X, Y, Z = self.grid
        
        # Получаем параметры из конфигурации
        nb_config = self.config['nanobridge']
        electrode_config = self.config['electrode']
        
        (_, center_box, _) = self.nano_system.nano_bridge.get_boxes()
        bridge_center = center_box.GetCenter()
        
        bridge_length = nb_config['grip_length']
        bridge_width = nb_config['grip_width']
        bridge_height = nb_config['grip_height']
        
        # Создаем маску для наномостика под электродом
        self.nanobridge_mask = np.zeros(X.shape, dtype=bool)
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                for k in range(X.shape[2]):
                    x, y, z = X[i,j,k], Y[i,j,k], Z[i,j,k]
                    
                    # Проверяем, находится ли точка внутри наномостика
                    in_bridge_x = abs(x - bridge_center.x) <= bridge_length/2
                    in_bridge_y = abs(y - bridge_center.y) <= bridge_width/2  
                    in_bridge_z = (z >= bridge_center.z) and (z <= bridge_height)

                    
                    if in_bridge_x and in_bridge_y and in_bridge_z:
                        self.nanobridge_mask[i,j,k] = True
        
        print(f"Найдено точек в наномостике под электродом: {np.sum(self.nanobridge_mask):,}")
        return True
    
    def extract_nanobridge_potential(self):
        """Извлекает потенциал только внутри наномостика под электродом"""
        if self.nanobridge_mask is None:
            self.create_nanobridge_mask()
            
        X, Y, Z = self.grid
        self.nanobridge_points = []
        self.nanobridge_potentials = []
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                for k in range(X.shape[2]):
                    if self.nanobridge_mask[i,j,k]:
                        self.nanobridge_points.append([X[i,j,k], Y[i,j,k], Z[i,j,k]])
                        self.nanobridge_potentials.append(self.potential[i,j,k])
        
        self.nanobridge_points = np.array(self.nanobridge_points)
        self.nanobridge_potentials = np.array(self.nanobridge_potentials)
        
        print(f"Извлечено точек: {len(self.nanobridge_points)}")
        if len(self.nanobridge_potentials) > 0:
            print(f"Диапазон потенциалов в наномостике: {self.nanobridge_potentials.min():.3f} - {self.nanobridge_potentials.max():.3f} V")
        
        return len(self.nanobridge_points) > 0
    
    def plot_potential_slices(self, save_plot=False):
        """Строит 2D срезы потенциала через наномостик"""
        if not self.extract_nanobridge_potential():
            return
            
        points = self.nanobridge_points
        potentials = self.nanobridge_potentials
        
        # Получаем параметры визуализации из конфига
        viz_config = self.config['visualization']
        slice_tolerance = viz_config['slice_tolerance']
        plot_style = viz_config['plot_style']
        
        fig, axes = plt.subplots(2, 3, figsize=plot_style['figure_size'])
        fig.suptitle('Распределение потенциала в наномостике под электродом', 
                    fontsize=plot_style['font_size'] + 2)
        
        # 1. Срез XY через центр по Z
        z_center = np.median(points[:, 2])
        mask_xy = (points[:, 2] >= z_center - slice_tolerance) & (points[:, 2] <= z_center + slice_tolerance)
        
        ax = axes[0, 0]
        sc = ax.scatter(points[mask_xy, 0], points[mask_xy, 1], c=potentials[mask_xy], 
                       cmap=viz_config['color_maps']['potential'], s=20, alpha=viz_config['opacity'])
        ax.set_xlabel('X (нм)')
        ax.set_ylabel('Y (нм)')
        ax.set_title(f'Срез XY (Z={z_center:.1f} нм)')
        ax.grid(True, alpha=0.3)
        plt.colorbar(sc, ax=ax, label='Потенциал (V)')
        
        # 2. Срез XZ через центр по Y
        y_center = np.median(points[:, 1])
        mask_xz = (points[:, 1] >= y_center - slice_tolerance) & (points[:, 1] <= y_center + slice_tolerance)
        
        ax = axes[0, 1]
        sc = ax.scatter(points[mask_xz, 0], points[mask_xz, 2], c=potentials[mask_xz], 
                       cmap=viz_config['color_maps']['electric_field'], s=20, alpha=viz_config['opacity'])
        ax.set_xlabel('X (нм)')
        ax.set_ylabel('Z (нм)')
        ax.set_title(f'Срез XZ (Y={y_center:.1f} нм)')
        ax.grid(True, alpha=0.3)
        plt.colorbar(sc, ax=ax, label='Потенциал (V)')
        
        # 3. Срез YZ через центр по X
        x_center = np.median(points[:, 0])
        mask_yz = (points[:, 0] >= x_center - slice_tolerance) & (points[:, 0] <= x_center + slice_tolerance)
        
        ax = axes[0, 2]
        sc = ax.scatter(points[mask_yz, 1], points[mask_yz, 2], c=potentials[mask_yz], 
                       cmap='coolwarm', s=20, alpha=viz_config['opacity'])
        ax.set_xlabel('Y (нм)')
        ax.set_ylabel('Z (нм)')
        ax.set_title(f'Срез YZ (X={x_center:.1f} нм)')
        ax.grid(True, alpha=0.3)
        plt.colorbar(sc, ax=ax, label='Потенциал (V)')
        
        # 4. Профиль потенциала вдоль X (в центре)
        ax = axes[1, 0]
        center_mask = (abs(points[:, 1] - y_center) < 1.0) & (abs(points[:, 2] - z_center) < 1.0)
        if np.any(center_mask):
            sorted_indices = np.argsort(points[center_mask, 0])
            ax.plot(points[center_mask, 0][sorted_indices], potentials[center_mask][sorted_indices], 
                   'b-', linewidth=2, marker='o', markersize=3)
            ax.set_xlabel('X (нм)')
            ax.set_ylabel('Потенциал (V)')
            ax.set_title('Профиль потенциала вдоль оси X\n(через центр наномостика)')
            ax.grid(True, alpha=0.3)
        
        # 5. Профиль потенциала вдоль Y (в центре)
        ax = axes[1, 1]
        center_mask = (abs(points[:, 0] - x_center) < 1.0) & (abs(points[:, 2] - z_center) < 1.0)
        if np.any(center_mask):
            sorted_indices = np.argsort(points[center_mask, 1])
            ax.plot(points[center_mask, 1][sorted_indices], potentials[center_mask][sorted_indices], 
                   'r-', linewidth=2, marker='s', markersize=3)
            ax.set_xlabel('Y (нм)')
            ax.set_ylabel('Потенциал (V)')
            ax.set_title('Профиль потенциала вдоль оси Y\n(через центр наномостика)')
            ax.grid(True, alpha=0.3)
        
        # 6. Гистограмма распределения потенциалов
        ax = axes[1, 2]
        ax.hist(potentials, bins=30, alpha=0.7, color='green', edgecolor='black')
        ax.set_xlabel('Потенциал (V)')
        ax.set_ylabel('Количество точек')
        ax.set_title('Распределение потенциалов в наномостике')
        ax.grid(True, alpha=0.3)
        
        # Добавляем статистику
        mean_pot = np.mean(potentials)
        std_pot = np.std(potentials)
        ax.axvline(mean_pot, color='red', linestyle='--', linewidth=2, 
                  label=f'Среднее: {mean_pot:.3f} V')
        ax.axvline(mean_pot + std_pot, color='orange', linestyle='--', alpha=0.7,
                  label=f'±σ: {std_pot:.3f} V')
        ax.axvline(mean_pot - std_pot, color='orange', linestyle='--', alpha=0.7)
        ax.legend()
        
        plt.tight_layout()
        
        if save_plot:
            output_path = os.path.join(self.config['file_paths']['output_directory'], 
                                     self.config['file_paths']['visualization_output'])
            plt.savefig(output_path, dpi=plot_style['dpi'])
            print(f"График сохранен в {output_path}")
        
        plt.show()
    
    def plot_3d_potential_volume(self):
        """3D визуализация потенциала в объеме наномостика"""
        if not self.extract_nanobridge_potential():
            return
            
        points = self.nanobridge_points
        potentials = self.nanobridge_potentials
        
        # Получаем параметры визуализации
        viz_config = self.config['visualization']
        
        # Создаем облако точек PyVista
        point_cloud = pv.PolyData(points)
        point_cloud["potential"] = potentials
        
        # Находим границы наномостика
        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()
        z_min, z_max = points[:, 2].min(), points[:, 2].max()
        
        print(f"Границы наномостика:")
        print(f"  X: {x_min:.2f} - {x_max:.2f} нм")
        print(f"  Y: {y_min:.2f} - {y_max:.2f} нм")
        print(f"  Z: {z_min:.2f} - {z_max:.2f} нм")
        
        # Создаем plotter с несколькими видами
        plotter = pv.Plotter(shape=(2, 2))
        
        # 1. 3D облако точек
        plotter.subplot(0, 0)
        plotter.add_mesh(point_cloud, scalars="potential", 
                        cmap=viz_config['color_maps']['potential'], 
                        point_size=viz_config['point_size'], 
                        opacity=viz_config['opacity'], 
                        render_points_as_spheres=True,
                        scalar_bar_args={'title': "Потенциал, V"})
        plotter.add_axes()
        plotter.add_title("3D распределение потенциала\nв наномостике")
        
        # 2. Срез через центр наномостика
        plotter.subplot(0, 1)
        
        # Создаем структурированную сетку для всего пространства
        X, Y, Z = self.grid
        grid = pv.StructuredGrid(X, Y, Z)
        grid["potential"] = self.potential.ravel(order='F')
        
        # Срез по YZ плоскости через центр
        center_x = (x_min + x_max) / 2
        slice_x = grid.slice(normal='x', origin=[center_x, 0, 0])
        plotter.add_mesh(slice_x, scalars="potential", 
                        cmap=viz_config['color_maps']['electric_field'], 
                        opacity=viz_config['opacity'],
                        scalar_bar_args={'title': "Потенциал, V"})
        plotter.add_title(f"Срез YZ (X={center_x:.1f} нм)")
        
        # 3. Срез по XZ плоскости через центр
        plotter.subplot(1, 0)
        center_y = (y_min + y_max) / 2
        slice_y = grid.slice(normal='y', origin=[0, center_y, 0])
        plotter.add_mesh(slice_y, scalars="potential", 
                        cmap='coolwarm', 
                        opacity=viz_config['opacity'],
                        scalar_bar_args={'title': "Потенциал, V"})
        plotter.add_title(f"Срез XZ (Y={center_y:.1f} нм)")
        
        # 4. Изоповерхности внутри наномостика
        plotter.subplot(1, 1)
        
        # Создаем ограничивающий бокс для наномостика
        nanobridge_bounds = [x_min, x_max, y_min, y_max, z_min, z_max]
        nanobridge_box = pv.Box(bounds=nanobridge_bounds)
        
        # Вырезаем область наномостика
        nanobridge_region = grid.clip_box(nanobridge_box, invert=False)
        
        if nanobridge_region.n_points > 0:
            # Создаем изоповерхности
            contours = nanobridge_region.contour(isosurfaces=8)
            plotter.add_mesh(contours, 
                           cmap=viz_config['color_maps']['electrode'], 
                           opacity=viz_config['opacity'],
                           scalar_bar_args={'title': "Потенциал, V"})
            plotter.add_mesh(nanobridge_box, color='white', opacity=0.1, 
                           show_edges=True, line_width=2)
            plotter.add_title("Изоповерхности потенциала\nв наномостике")
        
        plotter.show()
    
    def analyze_potential_well(self):
        """Детальный анализ потенциальной ямы в наномостике"""
        if not self.extract_nanobridge_potential():
            return
            
        points = self.nanobridge_points
        potentials = self.nanobridge_potentials
        
        # Получаем параметры анализа из конфига
        analysis_config = self.config['analysis']
        electrode_config = self.config['electrode']
        
        print("\n" + "="*60)
        print("ДЕТАЛЬНЫЙ АНАЛИЗ ПОТЕНЦИАЛЬНОЙ ЯМЫ В НАНОМОСТИКЕ")
        print("="*60)
        
        # Находим минимальный потенциал (дно ямы)
        min_potential_idx = np.argmin(potentials)
        min_potential = potentials[min_potential_idx]
        min_location = points[min_potential_idx]
        
        print(f"Минимальный потенциал (дно ямы): {min_potential:.6f} V")
        print(f"Местоположение минимума: X={min_location[0]:.2f}, Y={min_location[1]:.2f}, Z={min_location[2]:.2f} нм")
        
        # Глубина потенциальной ямы относительно электрода
        electrode_potential = electrode_config['potential']
        well_depth = electrode_potential - min_potential
        print(f"Глубина потенциальной ямы: {well_depth:.6f} V")
        
        # Размеры потенциальной ямы
        tolerance_percent = analysis_config['potential_tolerance_percent']
        tolerance = tolerance_percent / 100.0 * (potentials.max() - min_potential)
        well_points = points[potentials <= min_potential + tolerance]
        
        if len(well_points) > 0:
            well_size_x = well_points[:, 0].max() - well_points[:, 0].min()
            well_size_y = well_points[:, 1].max() - well_points[:, 1].min() 
            well_size_z = well_points[:, 2].max() - well_points[:, 2].min()
            
            print(f"\nРазмеры потенциальной ямы (по точкам в пределах {tolerance_percent}% от минимума):")
            print(f"  По X: {well_size_x:.2f} нм")
            print(f"  По Y: {well_size_y:.2f} нм")
            print(f"  По Z: {well_size_z:.2f} нм")
            print(f"  Объем: {well_size_x * well_size_y * well_size_z:.2f} нм³")
        
        # Градиент потенциала в центре ямы
        center_x, center_y, center_z = min_location
        gradient_radius = analysis_config['gradient_estimation_radius']
        
        # Находим точки вблизи центра для оценки градиента
        near_center_mask = (
            (abs(points[:, 0] - center_x) < gradient_radius) &
            (abs(points[:, 1] - center_y) < gradient_radius) & 
            (abs(points[:, 2] - center_z) < gradient_radius)
        )
        
        if np.sum(near_center_mask) > 10:
            near_points = points[near_center_mask]
            near_potentials = potentials[near_center_mask]
            
            # Оцениваем градиент по разностям
            grad_x = np.gradient(near_potentials, near_points[:, 0])
            grad_y = np.gradient(near_potentials, near_points[:, 1])
            grad_z = np.gradient(near_potentials, near_points[:, 2])
            
            avg_grad_x = np.mean(np.abs(grad_x))
            avg_grad_y = np.mean(np.abs(grad_y)) 
            avg_grad_z = np.mean(np.abs(grad_z))
            
            print(f"\nСредний градиент потенциала вблизи дна ямы:")
            print(f"  По X: {avg_grad_x:.6f} V/нм")
            print(f"  По Y: {avg_grad_y:.6f} V/нм")
            print(f"  По Z: {avg_grad_z:.6f} V/нм")
        
        # Симметрия потенциальной ямы
        x_center = np.mean(points[:, 0])
        left_mask = points[:, 0] < x_center
        right_mask = points[:, 0] > x_center
        
        if np.any(left_mask) and np.any(right_mask):
            left_avg = np.mean(potentials[left_mask])
            right_avg = np.mean(potentials[right_mask])
            symmetry_x = 1 - abs(left_avg - right_avg) / ((left_avg + right_avg) / 2)
            
            print(f"\nСимметрия потенциальной ямы:")
            print(f"  Средний потенциал слева: {left_avg:.6f} V")
            print(f"  Средний потенциал справа: {right_avg:.6f} V")
            print(f"  Коэффициент симметрии по X: {symmetry_x:.4f}")
        
        return min_potential, well_depth
    
    def quick_visualization(self, save_plots=True):
        """Быстрая комплексная визуализация"""
        if not self.load_data():
            return
            
        print("\n1. Анализ потенциальной ямы:")
        min_pot, depth = self.analyze_potential_well()
        
        print("\n2. Построение 2D срезов:")
        self.plot_potential_slices(save_plot=save_plots)
        
        print("\n3. 3D визуализация:")
        self.plot_3d_potential_volume()
        
        return min_pot, depth

def visualize_nanobridge_potential(config, nano_system, save_plots=True):
    visualizer = NanobridgePotentialVisualizer(config, nano_system)
    return visualizer.quick_visualization(save_plots=save_plots)
