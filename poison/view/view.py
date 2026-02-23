import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

class NanobridgePotentialVisualizer:    
    def __init__(self, config, nano_system):
        self.config = config
        self.nano_system = nano_system

        file_path = self.config['file_path']
        self.pkl_file = file_path['results_file']

        self.data = None
        self.grid = None
        self.potential = None
        self.electric_field = None

        output_dir = file_path.get('output_directory', 'results')
        os.makedirs(output_dir, exist_ok=True)
    
    def load_data(self):
        if not os.path.exists(self.pkl_file):
            print(f"Файл {self.pkl_file} не найден!")
            return False
            
        try:
            with open(self.pkl_file, 'rb') as f:
                self.data = pickle.load(f)
            
            self.potential = self.data['potential']
            self.grid = self.data['grid']
            self.electric_field = self.data.get('electric_field', None)
            
            print(f"Данные успешно загружены из {self.pkl_file}")
            print(f"Размер сетки: {self.potential.shape}")
            print(f"Диапазон потенциалов: {self.potential.min():.3f} - {self.potential.max():.3f} V")
            
            if self.electric_field is not None:
                Ex, Ey, Ez = self.electric_field
                print(f"Электрическое поле загружено:")
                print(f"  Ex: [{Ex.min():.3e}, {Ex.max():.3e}] V/нм")
                print(f"  Ey: [{Ey.min():.3e}, {Ey.max():.3e}] V/нм")
                print(f"  Ez: [{Ez.min():.3e}, {Ez.max():.3e}] V/нм")
            
            return True
            
        except Exception as e:
            print(f"Ошибка при загрузке файла: {e}")
            return False
    
    def plot_potential_profiles(self, save_plot=True):
        """Строит профили потенциала вдоль осей X, Y, Z через центр наномостика"""
        if self.grid is None or self.potential is None:
            print("Сначала загрузите данные!")
            return
            
        X, Y, Z = self.grid
        
        # Находим центральные индексы
        nx, ny, nz = self.potential.shape
        i_center = nx // 2
        j_center = ny // 2
        k_center = nz // 2
        
        # Координаты центра
        x_center = X[i_center, j_center, k_center]
        y_center = Y[i_center, j_center, k_center]
        z_center = Z[i_center, j_center, k_center]
        
        print(f"\nПостроение профилей потенциала через центр сетки:")
        print(f"Центральные индексы: i={i_center}, j={j_center}, k={k_center}")
        print(f"Координаты центра: X={x_center:.2f}, Y={y_center:.2f}, Z={z_center:.2f} нм")
        
        # Получаем параметры наномостика из конфига
        nb_config = self.config['nanobridge']
        grip_length = nb_config['grip_length']
        grip_width = nb_config['grip_width']
        end_length = nb_config['end_length']
        end_width = nb_config['end_width']
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle(f'Профили потенциала через центр наномостика\n(Y={y_center:.1f} нм, Z={z_center:.1f} нм для X; X={x_center:.1f} нм, Z={z_center:.1f} нм для Y; X={x_center:.1f} нм, Y={y_center:.1f} нм для Z)', 
                     fontsize=14, fontweight='bold')
        
        # ========== Профиль вдоль X (фиксированные j=j_center, k=k_center) ==========
        ax = axes[0]
        x_coords = X[:, j_center, k_center]
        x_potentials = self.potential[:, j_center, k_center]
        
        ax.plot(x_coords, x_potentials, 'b-', linewidth=2.5, marker='o', markersize=4)
        ax.set_xlabel('X (нм)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Потенциал (V)', fontsize=14, fontweight='bold')
        ax.set_title(f'Профиль вдоль X (Y={y_center:.1f} нм, Z={z_center:.1f} нм)', 
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(labelsize=11)
        
        # Аннотации частей наномостика вдоль X
        y_min, y_max = ax.get_ylim()
        y_text = y_min + (y_max - y_min) * 0.95
        
        # Левый резервуар
        left_end = -grip_length/2 - end_length
        right_end = -grip_length/2
        ax.axvspan(left_end, right_end, alpha=0.15, color='orange', label='Левый резервуар')
        ax.text((left_end + right_end)/2, y_text, 'Левый\nрезервуар', 
                ha='center', va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='orange', alpha=0.3))
        
        # Центральная часть (grip)
        left_grip = -grip_length/2
        right_grip = grip_length/2
        ax.axvspan(left_grip, right_grip, alpha=0.15, color='green', label='Наномостик')
        ax.text(0, y_text, 'Наномостик\n(grip)', 
                ha='center', va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='green', alpha=0.3))
        
        # Правый резервуар
        left_end_r = grip_length/2
        right_end_r = grip_length/2 + end_length
        ax.axvspan(left_end_r, right_end_r, alpha=0.15, color='orange', label='Правый резервуар')
        ax.text((left_end_r + right_end_r)/2, y_text, 'Правый\nрезервуар', 
                ha='center', va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='orange', alpha=0.3))
        
        print(f"Профиль X: {len(x_coords)} точек, потенциал: [{x_potentials.min():.4f}, {x_potentials.max():.4f}] V")
        
        # ========== Профиль вдоль Y (фиксированные i=i_center, k=k_center) ==========
        ax = axes[1]
        y_coords = Y[i_center, :, k_center]
        y_potentials = self.potential[i_center, :, k_center]
        
        ax.plot(y_coords, y_potentials, 'r-', linewidth=2.5, marker='s', markersize=4)
        ax.set_xlabel('Y (нм)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Потенциал (V)', fontsize=14, fontweight='bold')
        ax.set_title(f'Профиль вдоль Y (X={x_center:.1f} нм, Z={z_center:.1f} нм)', 
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(labelsize=11)
        
        # Аннотации вдоль Y (ширина наномостика)
        y_min_ax, y_max_ax = ax.get_ylim()
        y_text_y = y_min_ax + (y_max_ax - y_min_ax) * 0.95
        
        # Центральная часть наномостика
        ax.axvspan(-grip_width/2, grip_width/2, alpha=0.15, color='green')
        ax.text(0, y_text_y, 'Наномостик', 
                ha='center', va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='green', alpha=0.3))
        
        # Области вне наномостика
        y_min_coord = y_coords.min()
        y_max_coord = y_coords.max()
        if y_min_coord < -grip_width/2:
            ax.axvspan(y_min_coord, -grip_width/2, alpha=0.1, color='gray')
            ax.text((y_min_coord - grip_width/2)/2, y_text_y, 'Оксид/Воздух', 
                    ha='center', va='top', fontsize=8, bbox=dict(boxstyle='round', facecolor='gray', alpha=0.2))
        if y_max_coord > grip_width/2:
            ax.axvspan(grip_width/2, y_max_coord, alpha=0.1, color='gray')
            ax.text((y_max_coord + grip_width/2)/2, y_text_y, 'Оксид/Воздух', 
                    ha='center', va='top', fontsize=8, bbox=dict(boxstyle='round', facecolor='gray', alpha=0.2))
        
        print(f"Профиль Y: {len(y_coords)} точек, потенциал: [{y_potentials.min():.4f}, {y_potentials.max():.4f}] V")
        
        # ========== Профиль вдоль Z (фиксированные i=i_center, j=j_center) ==========
        ax = axes[2]
        z_coords = Z[i_center, j_center, :]
        z_potentials = self.potential[i_center, j_center, :]
        
        ax.plot(z_coords, z_potentials, 'g-', linewidth=2.5, marker='^', markersize=4)
        ax.set_xlabel('Z (нм)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Потенциал (V)', fontsize=14, fontweight='bold')
        ax.set_title(f'Профиль вдоль Z (X={x_center:.1f} нм, Y={y_center:.1f} нм)', 
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(labelsize=11)
        
        # Аннотации вдоль Z (высота структуры)
        y_min_z, y_max_z = ax.get_ylim()
        y_text_z = y_min_z + (y_max_z - y_min_z) * 0.95
        
        grip_height = nb_config['grip_height']
        
        # Подложка (ниже 0)
        z_min_coord = z_coords.min()
        if z_min_coord < 0:
            ax.axvspan(z_min_coord, 0, alpha=0.15, color='brown')
            ax.text((z_min_coord + 0)/2, y_text_z, 'Подложка', 
                    ha='center', va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='brown', alpha=0.3))
        
        # Наномостик (0 до grip_height)
        ax.axvspan(0, grip_height, alpha=0.15, color='green')
        ax.text(grip_height/2, y_text_z, 'Наномостик', 
                ha='center', va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='green', alpha=0.3))
        
        # Воздух/Электрод (выше grip_height)
        z_max_coord = z_coords.max()
        if z_max_coord > grip_height:
            ax.axvspan(grip_height, z_max_coord, alpha=0.1, color='cyan')
            ax.text((grip_height + z_max_coord)/2, y_text_z, 'Воздух/\nЭлектрод', 
                    ha='center', va='top', fontsize=8, bbox=dict(boxstyle='round', facecolor='cyan', alpha=0.3))
        
        print(f"Профиль Z: {len(z_coords)} точек, потенциал: [{z_potentials.min():.4f}, {z_potentials.max():.4f}] V")
        
        plt.tight_layout()
        
        if save_plot:
            output_path = os.path.join(self.config['file_path']['output_directory'],
                                     'potential_profiles.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"\nГрафик потенциала сохранен в {output_path}")
        
        plt.show()
    
    def plot_nanobridge_zoom_profiles(self, save_plot=True):
        """Строит детальные профили потенциала в области наномостика"""
        if self.grid is None or self.potential is None:
            print("Сначала загрузите данные!")
            return
            
        X, Y, Z = self.grid
        
        # Находим центральные индексы
        nx, ny, nz = self.potential.shape
        i_center = nx // 2
        j_center = ny // 2
        k_center = nz // 2
        
        # Координаты центра
        x_center = X[i_center, j_center, k_center]
        y_center = Y[i_center, j_center, k_center]
        z_center = Z[i_center, j_center, k_center]
        
        # Получаем параметры наномостика из конфига
        nb_config = self.config['nanobridge']
        grip_length = nb_config['grip_length']
        grip_width = nb_config['grip_width']
        grip_height = nb_config['grip_height']
        oxide_thickness = nb_config.get('oxide_thickness', 0.3)
        
        print(f"\nПостроение детальных профилей в области наномостика:")
        print(f"Центр: X={x_center:.2f}, Y={y_center:.2f}, Z={z_center:.2f} нм")
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle('Детальные профили потенциала в области наномостика\n(увеличенный масштаб для анализа границ)', 
                     fontsize=14, fontweight='bold')
        
        # ========== Профиль вдоль X (только центральная часть) ==========
        ax = axes[0]
        x_coords = X[:, j_center, k_center]
        x_potentials = self.potential[:, j_center, k_center]
        
        # Ограничиваем область вокруг наномостика
        x_margin = grip_length * 0.3  # 30% запас с каждой стороны
        x_mask = (x_coords >= -grip_length/2 - x_margin) & (x_coords <= grip_length/2 + x_margin)
        
        ax.plot(x_coords[x_mask], x_potentials[x_mask], 'b-', linewidth=2.5, marker='o', markersize=5)
        ax.set_xlabel('X (нм)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Потенциал (V)', fontsize=14, fontweight='bold')
        ax.set_title(f'Профиль вдоль X (детально)\nY={y_center:.1f} нм, Z={z_center:.1f} нм', 
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(labelsize=11)
        
        # Аннотации границ
        y_min, y_max = ax.get_ylim()
        
        # Границы наномостика
        ax.axvline(-grip_length/2, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Граница grip')
        ax.axvline(grip_length/2, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.axvspan(-grip_length/2, grip_length/2, alpha=0.15, color='green', label='Наномостик (grip)')
        
        ax.legend(fontsize=10)
        
        # ========== Профиль вдоль Y (через центр наномостика) ==========
        ax = axes[1]
        y_coords = Y[i_center, :, k_center]
        y_potentials = self.potential[i_center, :, k_center]
        
        # Ограничиваем область вокруг наномостика
        y_margin = (grip_width + 2*oxide_thickness) * 1.5  # Включаем оксид + запас
        y_mask = (y_coords >= -y_margin) & (y_coords <= y_margin)
        
        ax.plot(y_coords[y_mask], y_potentials[y_mask], 'r-', linewidth=2.5, marker='s', markersize=5)
        ax.set_xlabel('Y (нм)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Потенциал (V)', fontsize=14, fontweight='bold')
        ax.set_title(f'Профиль вдоль Y (детально)\nX={x_center:.1f} нм, Z={z_center:.1f} нм', 
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(labelsize=11)
        
        # Аннотации границ
        y_min_ax, y_max_ax = ax.get_ylim()
        
        # Границы наномостика
        ax.axvline(-grip_width/2, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Граница Si')
        ax.axvline(grip_width/2, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.axvspan(-grip_width/2, grip_width/2, alpha=0.15, color='green', label='Наномостик (Si)')
        
        # Границы оксида
        oxide_inner = grip_width/2
        oxide_outer = grip_width/2 + oxide_thickness
        ax.axvline(-oxide_outer, color='orange', linestyle=':', linewidth=2, alpha=0.7, label='Граница оксида')
        ax.axvline(oxide_outer, color='orange', linestyle=':', linewidth=2, alpha=0.7)
        ax.axvspan(-oxide_outer, -oxide_inner, alpha=0.1, color='orange', label='Оксид')
        ax.axvspan(oxide_inner, oxide_outer, alpha=0.1, color='orange')
        
        ax.legend(fontsize=9, loc='best')
        
        # ========== Профиль вдоль Z (от подложки до электрода) ==========
        ax = axes[2]
        z_coords = Z[i_center, j_center, :]
        z_potentials = self.potential[i_center, j_center, :]
        
        # Ограничиваем область от подложки до чуть выше наномостика
        z_margin_below = 2.0  # нм ниже наномостика
        z_margin_above = grip_height * 0.5  # выше наномостика
        z_mask = (z_coords >= -z_margin_below) & (z_coords <= grip_height + z_margin_above)
        
        ax.plot(z_coords[z_mask], z_potentials[z_mask], 'g-', linewidth=2.5, marker='^', markersize=5)
        ax.set_xlabel('Z (нм)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Потенциал (V)', fontsize=14, fontweight='bold')
        ax.set_title(f'Профиль вдоль Z (детально)\nX={x_center:.1f} нм, Y={y_center:.1f} нм', 
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(labelsize=11)
        
        # Аннотации границ
        y_min_z, y_max_z = ax.get_ylim()
        
        # Граница подложка-наномостик
        ax.axvline(0, color='brown', linestyle='--', linewidth=2, alpha=0.7, label='Граница подложка-Si')
        ax.axvspan(z_coords[z_mask].min(), 0, alpha=0.15, color='brown', label='Подложка')
        
        # Граница наномостик-воздух/оксид
        ax.axvline(grip_height, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Граница Si-оксид')
        ax.axvspan(0, grip_height, alpha=0.15, color='green', label='Наномостик (Si)')
        
        # Область оксида/воздуха
        if grip_height + oxide_thickness < z_coords[z_mask].max():
            ax.axvline(grip_height + oxide_thickness, color='orange', linestyle=':', linewidth=2, alpha=0.7, label='Граница оксид-воздух')
            ax.axvspan(grip_height, grip_height + oxide_thickness, alpha=0.1, color='orange', label='Оксид')
        
        ax.legend(fontsize=9, loc='best')
        
        plt.tight_layout()
        
        if save_plot:
            output_path = os.path.join(self.config['file_path']['output_directory'], 
                                     'potential_profiles_zoom.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"\nДетальный график потенциала сохранен в {output_path}")
        
        plt.show()
    
    def plot_nanobridge_zoom_field_profiles(self, save_plot=True):
        """Строит детальные профили электрического поля в области наномостика"""
        if self.electric_field is None:
            print("Электрическое поле не загружено!")
            return
            
        Ex, Ey, Ez = self.electric_field
        X, Y, Z = self.grid
        
        # Находим центральные индексы
        nx, ny, nz = Ex.shape
        i_center = nx // 2
        j_center = ny // 2
        k_center = nz // 2
        
        # Координаты центра
        x_center = X[i_center, j_center, k_center]
        y_center = Y[i_center, j_center, k_center]
        z_center = Z[i_center, j_center, k_center]
        
        # Получаем параметры наномостика из конфига
        nb_config = self.config['nanobridge']
        grip_length = nb_config['grip_length']
        grip_width = nb_config['grip_width']
        grip_height = nb_config['grip_height']
        
        print(f"\nПостроение детальных профилей электрического поля в области наномостика:")
        
        fig, axes = plt.subplots(3, 3, figsize=(20, 16))
        fig.suptitle('Детальные профили электрического поля в области наномостика\n(увеличенный масштаб для анализа границ)', 
                     fontsize=14, fontweight='bold')
        
        # ========== Профили вдоль X (центральная часть) ==========
        x_coords = X[:, j_center, k_center]
        x_margin = grip_length * 0.3
        x_mask = (x_coords >= -grip_length/2 - x_margin) & (x_coords <= grip_length/2 + x_margin)
        
        # Ex вдоль X
        ax = axes[0, 0]
        ex_profile = Ex[:, j_center, k_center]
        ax.plot(x_coords[x_mask], ex_profile[x_mask], 'b-', linewidth=2.5, marker='o', markersize=4)
        ax.set_ylabel('Ex (V/нм)', fontsize=12, fontweight='bold')
        ax.set_title('Ex вдоль X (детально)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax.axvline(-grip_length/2, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
        ax.axvline(grip_length/2, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
        ax.axvspan(-grip_length/2, grip_length/2, alpha=0.1, color='green')
        
        # Ey вдоль X
        ax = axes[1, 0]
        ey_profile = Ey[:, j_center, k_center]
        ax.plot(x_coords[x_mask], ey_profile[x_mask], 'r-', linewidth=2.5, marker='s', markersize=4)
        ax.set_ylabel('Ey (V/нм)', fontsize=12, fontweight='bold')
        ax.set_title('Ey вдоль X (детально)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax.axvline(-grip_length/2, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
        ax.axvline(grip_length/2, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
        ax.axvspan(-grip_length/2, grip_length/2, alpha=0.1, color='green')
        
        # Ez вдоль X
        ax = axes[2, 0]
        ez_profile = Ez[:, j_center, k_center]
        ax.plot(x_coords[x_mask], ez_profile[x_mask], 'g-', linewidth=2.5, marker='^', markersize=4)
        ax.set_xlabel('X (нм)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Ez (V/нм)', fontsize=12, fontweight='bold')
        ax.set_title('Ez вдоль X (детально)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax.axvline(-grip_length/2, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
        ax.axvline(grip_length/2, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
        ax.axvspan(-grip_length/2, grip_length/2, alpha=0.1, color='green')
        
        # ========== Профили вдоль Y (область наномостика + оксид) ==========
        y_coords = Y[i_center, :, k_center]
        y_margin = (grip_width + 4) * 1.5
        y_mask = (y_coords >= -y_margin) & (y_coords <= y_margin)
        
        # Ex вдоль Y
        ax = axes[0, 1]
        ex_profile_y = Ex[i_center, :, k_center]
        ax.plot(y_coords[y_mask], ex_profile_y[y_mask], 'b-', linewidth=2.5, marker='o', markersize=4)
        ax.set_ylabel('Ex (V/нм)', fontsize=12, fontweight='bold')
        ax.set_title('Ex вдоль Y (детально)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax.axvline(-grip_width/2, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
        ax.axvline(grip_width/2, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
        ax.axvspan(-grip_width/2, grip_width/2, alpha=0.1, color='green')
        
        # Ey вдоль Y
        ax = axes[1, 1]
        ey_profile_y = Ey[i_center, :, k_center]
        ax.plot(y_coords[y_mask], ey_profile_y[y_mask], 'r-', linewidth=2.5, marker='s', markersize=4)
        ax.set_ylabel('Ey (V/нм)', fontsize=12, fontweight='bold')
        ax.set_title('Ey вдоль Y (детально)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax.axvline(-grip_width/2, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
        ax.axvline(grip_width/2, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
        ax.axvspan(-grip_width/2, grip_width/2, alpha=0.1, color='green')
        
        # Ez вдоль Y
        ax = axes[2, 1]
        ez_profile_y = Ez[i_center, :, k_center]
        ax.plot(y_coords[y_mask], ez_profile_y[y_mask], 'g-', linewidth=2.5, marker='^', markersize=4)
        ax.set_xlabel('Y (нм)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Ez (V/нм)', fontsize=12, fontweight='bold')
        ax.set_title('Ez вдоль Y (детально)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax.axvline(-grip_width/2, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
        ax.axvline(grip_width/2, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
        ax.axvspan(-grip_width/2, grip_width/2, alpha=0.1, color='green')
        
        # ========== Профили вдоль Z (от подложки до электрода) ==========
        z_coords = Z[i_center, j_center, :]
        z_margin_below = 2.0
        z_margin_above = grip_height * 0.5
        z_mask = (z_coords >= -z_margin_below) & (z_coords <= grip_height + z_margin_above)
        
        # Ex вдоль Z
        ax = axes[0, 2]
        ex_profile_z = Ex[i_center, j_center, :]
        ax.plot(z_coords[z_mask], ex_profile_z[z_mask], 'b-', linewidth=2.5, marker='o', markersize=4)
        ax.set_ylabel('Ex (V/нм)', fontsize=12, fontweight='bold')
        ax.set_title('Ex вдоль Z (детально)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax.axvline(0, color='brown', linestyle='--', linewidth=1.5, alpha=0.5)
        ax.axvline(grip_height, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
        ax.axvspan(0, grip_height, alpha=0.1, color='green')
        
        # Ey вдоль Z
        ax = axes[1, 2]
        ey_profile_z = Ey[i_center, j_center, :]
        ax.plot(z_coords[z_mask], ey_profile_z[z_mask], 'r-', linewidth=2.5, marker='s', markersize=4)
        ax.set_ylabel('Ey (V/нм)', fontsize=12, fontweight='bold')
        ax.set_title('Ey вдоль Z (детально)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax.axvline(0, color='brown', linestyle='--', linewidth=1.5, alpha=0.5)
        ax.axvline(grip_height, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
        ax.axvspan(0, grip_height, alpha=0.1, color='green')
        
        # Ez вдоль Z
        ax = axes[2, 2]
        ez_profile_z = Ez[i_center, j_center, :]
        ax.plot(z_coords[z_mask], ez_profile_z[z_mask], 'g-', linewidth=2.5, marker='^', markersize=4)
        ax.set_xlabel('Z (нм)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Ez (V/нм)', fontsize=12, fontweight='bold')
        ax.set_title('Ez вдоль Z (детально)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax.axvline(0, color='brown', linestyle='--', linewidth=1.5, alpha=0.5)
        ax.axvline(grip_height, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
        ax.axvspan(0, grip_height, alpha=0.1, color='green')
        
        plt.tight_layout()
        
        if save_plot:
            output_path = os.path.join(self.config['file_path']['output_directory'], 
                                     'electric_field_profiles_zoom.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"\nДетальный график электрического поля сохранен в {output_path}")
        
        plt.show()
    
    def plot_electric_field_profiles(self, save_plot=True):
        """Строит профили компонент электрического поля вдоль осей X, Y, Z"""
        if self.electric_field is None:
            print("Электрическое поле не загружено!")
            return
            
        Ex, Ey, Ez = self.electric_field
        X, Y, Z = self.grid
        
        # Находим центральные индексы
        nx, ny, nz = Ex.shape
        i_center = nx // 2
        j_center = ny // 2
        k_center = nz // 2
        
        # Координаты центра
        x_center = X[i_center, j_center, k_center]
        y_center = Y[i_center, j_center, k_center]
        z_center = Z[i_center, j_center, k_center]
        
        print(f"\nПостроение профилей электрического поля через центр сетки:")
        print(f"Координаты центра: X={x_center:.2f}, Y={y_center:.2f}, Z={z_center:.2f} нм")
        
        # Получаем параметры наномостика из конфига
        nb_config = self.config['nanobridge']
        grip_length = nb_config['grip_length']
        grip_width = nb_config['grip_width']
        grip_height = nb_config['grip_height']
        
        fig, axes = plt.subplots(3, 3, figsize=(20, 16))
        fig.suptitle(f'Профили электрического поля через центр наномостика\n(Y={y_center:.1f} нм, Z={z_center:.1f} нм для X; X={x_center:.1f} нм, Z={z_center:.1f} нм для Y; X={x_center:.1f} нм, Y={y_center:.1f} нм для Z)', 
                     fontsize=14, fontweight='bold')
        
        # ========== Профили вдоль X ==========
        x_coords = X[:, j_center, k_center]
        
        # Ex вдоль X
        ax = axes[0, 0]
        ex_profile = Ex[:, j_center, k_center]
        ax.plot(x_coords, ex_profile, 'b-', linewidth=2.5, marker='o', markersize=3)
        ax.set_ylabel('Ex (V/нм)', fontsize=12, fontweight='bold')
        ax.set_title('Ex вдоль X', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axhline(0, color='k', linestyle='--', alpha=0.3)
        self._add_x_annotations(ax, grip_length)
        
        # Ey вдоль X
        ax = axes[1, 0]
        ey_profile = Ey[:, j_center, k_center]
        ax.plot(x_coords, ey_profile, 'r-', linewidth=2.5, marker='s', markersize=3)
        ax.set_ylabel('Ey (V/нм)', fontsize=12, fontweight='bold')
        ax.set_title('Ey вдоль X', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axhline(0, color='k', linestyle='--', alpha=0.3)
        self._add_x_annotations(ax, grip_length)
        
        # Ez вдоль X
        ax = axes[2, 0]
        ez_profile = Ez[:, j_center, k_center]
        ax.plot(x_coords, ez_profile, 'g-', linewidth=2.5, marker='^', markersize=3)
        ax.set_xlabel('X (нм)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Ez (V/нм)', fontsize=12, fontweight='bold')
        ax.set_title('Ez вдоль X', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axhline(0, color='k', linestyle='--', alpha=0.3)
        self._add_x_annotations(ax, grip_length)
        
        # ========== Профили вдоль Y ==========
        y_coords = Y[i_center, :, k_center]
        
        # Ex вдоль Y
        ax = axes[0, 1]
        ex_profile_y = Ex[i_center, :, k_center]
        ax.plot(y_coords, ex_profile_y, 'b-', linewidth=2.5, marker='o', markersize=3)
        ax.set_ylabel('Ex (V/нм)', fontsize=12, fontweight='bold')
        ax.set_title('Ex вдоль Y', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axhline(0, color='k', linestyle='--', alpha=0.3)
        self._add_y_annotations(ax, grip_width)
        
        # Ey вдоль Y
        ax = axes[1, 1]
        ey_profile_y = Ey[i_center, :, k_center]
        ax.plot(y_coords, ey_profile_y, 'r-', linewidth=2.5, marker='s', markersize=3)
        ax.set_ylabel('Ey (V/нм)', fontsize=12, fontweight='bold')
        ax.set_title('Ey вдоль Y', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axhline(0, color='k', linestyle='--', alpha=0.3)
        self._add_y_annotations(ax, grip_width)
        
        # Ez вдоль Y
        ax = axes[2, 1]
        ez_profile_y = Ez[i_center, :, k_center]
        ax.plot(y_coords, ez_profile_y, 'g-', linewidth=2.5, marker='^', markersize=3)
        ax.set_xlabel('Y (нм)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Ez (V/нм)', fontsize=12, fontweight='bold')
        ax.set_title('Ez вдоль Y', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axhline(0, color='k', linestyle='--', alpha=0.3)
        self._add_y_annotations(ax, grip_width)
        
        # ========== Профили вдоль Z ==========
        z_coords = Z[i_center, j_center, :]
        
        # Ex вдоль Z
        ax = axes[0, 2]
        ex_profile_z = Ex[i_center, j_center, :]
        ax.plot(z_coords, ex_profile_z, 'b-', linewidth=2.5, marker='o', markersize=3)
        ax.set_ylabel('Ex (V/нм)', fontsize=12, fontweight='bold')
        ax.set_title('Ex вдоль Z', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axhline(0, color='k', linestyle='--', alpha=0.3)
        self._add_z_annotations(ax, grip_height)
        
        # Ey вдоль Z
        ax = axes[1, 2]
        ey_profile_z = Ey[i_center, j_center, :]
        ax.plot(z_coords, ey_profile_z, 'r-', linewidth=2.5, marker='s', markersize=3)
        ax.set_ylabel('Ey (V/нм)', fontsize=12, fontweight='bold')
        ax.set_title('Ey вдоль Z', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axhline(0, color='k', linestyle='--', alpha=0.3)
        self._add_z_annotations(ax, grip_height)
        # Проверка на малые значения
        if np.abs(ey_profile_z).max() < 1e-10:
            ax.text(0.5, 0.5, f'Ey ≈ 0\n(max: {np.abs(ey_profile_z).max():.2e})',
                   ha='center', va='center', transform=ax.transAxes, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
        
        # Ez вдоль Z
        ax = axes[2, 2]
        ez_profile_z = Ez[i_center, j_center, :]
        ax.plot(z_coords, ez_profile_z, 'g-', linewidth=2.5, marker='^', markersize=3)
        ax.set_xlabel('Z (нм)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Ez (V/нм)', fontsize=12, fontweight='bold')
        ax.set_title('Ez вдоль Z', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axhline(0, color='k', linestyle='--', alpha=0.3)
        self._add_z_annotations(ax, grip_height)
        # Проверка на малые значения
        if np.abs(ez_profile_z).max() < 1e-10:
            ax.text(0.5, 0.5, f'Ez ≈ 0\n(max: {np.abs(ez_profile_z).max():.2e})',
                   ha='center', va='center', transform=ax.transAxes, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
        
        # Выводим статистику
        print(f"\nСтатистика электрического поля:")
        print(f"  Ex вдоль X: [{ex_profile.min():.3e}, {ex_profile.max():.3e}] V/нм")
        print(f"  Ey вдоль Y: [{ey_profile_y.min():.3e}, {ey_profile_y.max():.3e}] V/нм")
        print(f"  Ez вдоль Z: [{ez_profile_z.min():.3e}, {ez_profile_z.max():.3e}] V/нм")
        print(f"  Ey вдоль Z: [{ey_profile_z.min():.3e}, {ey_profile_z.max():.3e}] V/нм (может быть ≈0)")
        
        plt.tight_layout()
        
        if save_plot:
            output_path = os.path.join(self.config['file_path']['output_directory'], 
                                     'electric_field_profiles.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"\nГрафик электрического поля сохранен в {output_path}")
        
        plt.show()
    
    def _add_x_annotations(self, ax, grip_length):
        """Добавляет аннотации для профилей вдоль X"""
        y_min, y_max = ax.get_ylim()
        # Наномостик в центре
        ax.axvspan(-grip_length/2, grip_length/2, alpha=0.1, color='green')
    
    def _add_y_annotations(self, ax, grip_width):
        """Добавляет аннотации для профилей вдоль Y"""
        y_min, y_max = ax.get_ylim()
        # Наномостик в центре
        ax.axvspan(-grip_width/2, grip_width/2, alpha=0.1, color='green')
    
    def _add_z_annotations(self, ax, grip_height):
        """Добавляет аннотации для профилей вдоль Z"""
        y_min, y_max = ax.get_ylim()
        # Наномостик от 0 до grip_height
        ax.axvspan(0, grip_height, alpha=0.1, color='green')
    
    def quick_visualization(self, save_plots=True, include_zoom=True):
        """Быстрая визуализация - профили потенциала и электрического поля"""
        if not self.load_data():
            return
        
        print("\n" + "="*60)
        print("ВИЗУАЛИЗАЦИЯ ПОТЕНЦИАЛА (полный масштаб)")
        print("="*60)
        self.plot_potential_profiles(save_plot=save_plots)
        
        if include_zoom:
            print("\n" + "="*60)
            print("ДЕТАЛЬНАЯ ВИЗУАЛИЗАЦИЯ ПОТЕНЦИАЛА (область наномостика)")
            print("="*60)
            self.plot_nanobridge_zoom_profiles(save_plot=save_plots)
        
        if self.electric_field is not None:
            print("\n" + "="*60)
            print("ВИЗУАЛИЗАЦИЯ ЭЛЕКТРИЧЕСКОГО ПОЛЯ (полный масштаб)")
            print("="*60)
            self.plot_electric_field_profiles(save_plot=save_plots)
            
            if include_zoom:
                print("\n" + "="*60)
                print("ДЕТАЛЬНАЯ ВИЗУАЛИЗАЦИЯ ЭЛЕКТРИЧЕСКОГО ПОЛЯ (область наномостика)")
                print("="*60)
                self.plot_nanobridge_zoom_field_profiles(save_plot=save_plots)
        else:
            print("\nВНИМАНИЕ: Электрическое поле не найдено в данных!")

def visualize_nanobridge_potential(config, nano_system, save_plots=True, include_zoom=True):
    """Главная функция для визуализации потенциала в наномостике
    
    Args:
        config: Конфигурация системы
        nano_system: Модель наносистемы
        save_plots: Сохранять ли графики (по умолчанию True)
        include_zoom: Включать ли детальные графики области наномостика (по умолчанию True)
    """
    visualizer = NanobridgePotentialVisualizer(config, nano_system)
    return visualizer.quick_visualization(save_plots=save_plots, include_zoom=include_zoom)
