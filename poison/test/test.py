import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from typing import List

class QuickFieldAnalyzer:
    """Быстрый анализатор полей из сохраненных файлов"""
    
    def __init__(self, filename="data/electric_field_results.pkl"):
        self.filename = filename
        self.data = None
        self.load_data()
    
    def load_data(self):
        """Загружает данные из файла"""
        try:
            with open(self.filename, 'rb') as f:
                self.data = pickle.load(f)
            print(f"✓ Файл {self.filename} загружен успешно")
            print(f"  Размер сетки: {self.data['potential'].shape}")
            print(f"  Дата расчета: {self.data.get('timestamp', 'неизвестно')}")
            return True
        except Exception as e:
            print(f"✗ Ошибка загрузки файла: {e}")
            return False
    
    def test_boundary_conditions(self):
        """Тест граничных условий"""
        print("\n" + "="*50)
        print("ТЕСТ ГРАНИЧНЫХ УСЛОВИЙ")
        print("="*50)
        
        potential = self.data['potential']
        grid = self.data['grid']
        mask = self.data['mask']
        
        # 1. Проверяем границы (должны быть близки к 0V)
        boundary_values = potential[mask & (potential < 0.1)]
        if len(boundary_values) > 0:
            boundary_mean = np.mean(boundary_values)
            print(f"Граничные точки (~0V): {len(boundary_values):,} точек")
            print(f"Средний потенциал на границах: {boundary_mean:.3f} V")
        
        # 2. Проверяем электрод (должен быть близок к 10V)
        electrode_values = potential[mask & (potential > 9.9)]
        if len(electrode_values) > 0:
            electrode_mean = np.mean(electrode_values)
            print(f"Точки электрода (~10V): {len(electrode_values):,} точек")
            print(f"Средний потенциал электрода: {electrode_mean:.3f} V")
        
        return len(boundary_values) > 0 and len(electrode_values) > 0
    
    def test_potential_well(self):
        """Тест наличия и характеристик потенциальной ямы"""
        print("\n" + "="*50)
        print("ТЕСТ ПОТЕНЦИАЛЬНОЙ ЯМЫ")
        print("="*50)
        
        potential = self.data['potential']
        X, Y, Z = self.data['grid']
        
        # Определяем область нанопровода (примерные координаты)
        # Центральная часть: X ~ [-10, 10], Y ~ [-4, 4], Z ~ [0, 6]
        nanowire_mask = (
            (X >= -15) & (X <= 15) & 
            (Y >= -6) & (Y <= 6) & 
            (Z >= 0) & (Z <= 8)
        )
        
        nanowire_potentials = potential[nanowire_mask]
        
        if len(nanowire_potentials) == 0:
            print("✗ Не найдены точки в области нанопровода")
            return False
        
        min_potential = np.min(nanowire_potentials)
        max_potential = np.max(nanowire_potentials)
        mean_potential = np.mean(nanowire_potentials)
        
        print(f"Потенциал в нанопроводе:")
        print(f"  Минимальный: {min_potential:.3f} V")
        print(f"  Максимальный: {max_potential:.3f} V")
        print(f"  Средний: {mean_potential:.3f} V")
        
        # Глубина потенциальной ямы относительно затвора
        gate_potential = 10.0
        well_depth = gate_potential - min_potential
        print(f"Глубина потенциальной ямы: {well_depth:.3f} V")
        
        # Ожидаемые характеристики
        expected_min = 2.0  # Ожидаемый минимум потенциала
        expected_max = 8.0  # Ожидаемый максимум в нанопроводе
        
        if min_potential < expected_min:
            print(f"✓ Глубокая потенциальная яма (< {expected_min}V)")
        else:
            print(f"⚠️  Мелкая потенциальная яма (≥ {expected_min}V)")
        
        if well_depth > 3.0:
            print("✓ Значительная глубина ямы (> 3V)")
        else:
            print("⚠️  Небольшая глубина ямы (≤ 3V)")
        
        return True
    
    def test_field_uniformity(self):
        """Тест однородности поля в электроде"""
        print("\n" + "="*50)
        print("ТЕСТ ОДНОРОДНОСТИ ПОЛЯ")
        print("="*50)
        
        potential = self.data['potential']
        X, Y, Z = self.data['grid']
        
        # Область электрода (примерные координаты)
        electrode_mask = (
            (X >= -12) & (X <= 12) & 
            (Y >= -8) & (Y <= 8) & 
            (Z >= 0) & (Z <= 10)
        )
        
        electrode_potentials = potential[electrode_mask]
        
        if len(electrode_potentials) == 0:
            print("✗ Не найдены точки в электроде")
            return False
        
        electrode_std = np.std(electrode_potentials)
        electrode_mean = np.mean(electrode_potentials)
        
        print(f"Потенциал в электроде:")
        print(f"  Средний: {electrode_mean:.3f} V")
        print(f"  Стандартное отклонение: {electrode_std:.3f} V")
        print(f"  Относительная неоднородность: {electrode_std/electrode_mean*100:.2f}%")
        
        # Для идеального проводника отклонение должно быть маленьким
        if electrode_std < 0.1:
            print("✓ Электрод близок к идеальному проводнику")
        elif electrode_std < 0.5:
            print("⚠️  Умеренная неоднородность в электроде")
        else:
            print("✗ Значительная неоднородность в электроде")
        
        return True
    
    def test_substrate_behavior(self):
        """Тест поведения подложки"""
        print("\n" + "="*50)
        print("ТЕСТ ПОВЕДЕНИЯ ПОДЛОЖКИ")
        print("="*50)
        
        potential = self.data['potential']
        X, Y, Z = self.data['grid']
        
        # Область подложки (нижняя часть)
        substrate_mask = (Z <= -2)
        substrate_potentials = potential[substrate_mask]
        
        if len(substrate_potentials) == 0:
            print("✗ Не найдены точки в подложке")
            return False
        
        substrate_mean = np.mean(substrate_potentials)
        substrate_std = np.std(substrate_potentials)
        
        print(f"Потенциал в подложке:")
        print(f"  Средний: {substrate_mean:.3f} V")
        print(f"  Стандартное отклонение: {substrate_std:.3f} V")
        print(f"  Диапазон: [{np.min(substrate_potentials):.3f}, {np.max(substrate_potentials):.3f}] V")
        
        # Проверяем, ведет ли себя подложка как диэлектрик
        if abs(substrate_mean) < 0.1 and substrate_std < 0.1:
            print("✗ Подложка ведет себя как проводник (постоянный 0V)")
            return False
        else:
            print("✓ Подложка ведет себя как диэлектрик (потенциал изменяется)")
            return True
    
    def plot_quick_analysis(self):
        """Быстрая визуализация ключевых срезов"""
        print("\n" + "="*50)
        print("БЫСТРАЯ ВИЗУАЛИЗАЦИЯ")
        print("="*50)
        
        potential = self.data['potential']
        X, Y, Z = self.data['grid']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Срез XZ через центр (Y=0)
        y_center_idx = np.argmin(np.abs(Y[0,:,0]))
        slice_xz = potential[:, y_center_idx, :]
        
        im1 = axes[0,0].imshow(slice_xz.T, extent=[X.min(), X.max(), Z.min(), Z.max()], 
                              origin='lower', cmap='coolwarm', aspect='auto')
        axes[0,0].set_xlabel('X (мкм)')
        axes[0,0].set_ylabel('Z (мкм)')
        axes[0,0].set_title('Срез XZ (Y=0)')
        plt.colorbar(im1, ax=axes[0,0], label='Потенциал (V)')
        
        # 2. Срез XY через середину высоты нанопровода (Z=3)
        z_mid_idx = np.argmin(np.abs(Z[0,0,:] - 3))
        slice_xy = potential[:, :, z_mid_idx]
        
        im2 = axes[0,1].imshow(slice_xy.T, extent=[X.min(), X.max(), Y.min(), Y.max()], 
                              origin='lower', cmap='coolwarm', aspect='auto')
        axes[0,1].set_xlabel('X (мкм)')
        axes[0,1].set_ylabel('Y (мкм)')
        axes[0,1].set_title('Срез XY (Z=3)')
        plt.colorbar(im2, ax=axes[0,1], label='Потенциал (V)')
        
        # 3. Профиль потенциала вдоль X через центр
        x_profile = potential[:, y_center_idx, z_mid_idx]
        axes[1,0].plot(X[:,0,0], x_profile, 'b-', linewidth=2)
        axes[1,0].set_xlabel('X (мкм)')
        axes[1,0].set_ylabel('Потенциал (V)')
        axes[1,0].set_title('Профиль вдоль X (через центр)')
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Гистограмма потенциалов в нанопроводе
        nanowire_mask = (
            (X >= -10) & (X <= 10) & 
            (Y >= -4) & (Y <= 4) & 
            (Z >= 1) & (Z <= 5)
        )
        nanowire_potentials = potential[nanowire_mask]
        
        axes[1,1].hist(nanowire_potentials, bins=30, alpha=0.7, color='green', edgecolor='black')
        axes[1,1].set_xlabel('Потенциал (V)')
        axes[1,1].set_ylabel('Количество точек')
        axes[1,1].set_title('Распределение в нанопроводе')
        axes[1,1].grid(True, alpha=0.3)
        
        # Добавляем статистику на гистограмму
        mean_pot = np.mean(nanowire_potentials)
        min_pot = np.min(nanowire_potentials)
        axes[1,1].axvline(mean_pot, color='red', linestyle='--', label=f'Среднее: {mean_pot:.2f}V')
        axes[1,1].axvline(min_pot, color='blue', linestyle=':', label=f'Мин: {min_pot:.2f}V')
        axes[1,1].legend()
        
        plt.tight_layout()
        plt.show()
    
    def run_comprehensive_analysis(self):
        """Запуск всестороннего анализа"""
        if self.data is None:
            print("Нет данных для анализа!")
            return False
        
        print("🚀 ЗАПУСК КОМПЛЕКСНОГО АНАЛИЗА РЕЗУЛЬТАТОВ")
        print("="*60)
        
        tests = [
            ("Граничные условия", self.test_boundary_conditions),
            ("Потенциальная яма", self.test_potential_well),
            ("Однородность поля", self.test_field_uniformity),
            ("Поведение подложки", self.test_substrate_behavior),
        ]
        
        results = []
        for test_name, test_func in tests:
            try:
                success = test_func()
                results.append((test_name, success))
                print(f"--- {test_name}: {'ПРОЙДЕН' if success else 'НЕ ПРОЙДЕН'} ---\n")
            except Exception as e:
                print(f"Ошибка в тесте '{test_name}': {e}")
                results.append((test_name, False))
        
        # Сводка
        print("📊 СВОДКА РЕЗУЛЬТАТОВ:")
        print("="*40)
        passed = sum(1 for _, success in results if success)
        total = len(results)
        
        for test_name, success in results:
            status = "✅ ПРОЙДЕН" if success else "❌ НЕ ПРОЙДЕН"
            print(f"{test_name}: {status}")
        
        print(f"\nИтог: {passed}/{total} тестов пройдено")
        
        if passed == total:
            print("🎉 Отличные результаты! Модель работает корректно.")
        elif passed >= total * 0.7:
            print("⚠️  Удовлетворительные результаты. Есть небольшие проблемы.")
        else:
            print("🔴 Критические проблемы в решении!")
        
        # Визуализация
        self.plot_quick_analysis()
        
        return passed == total

# Функции для быстрого запуска
def quick_analyze(filename="electric_field_results.pkl"):
    """Быстрый анализ файла с результатами"""
    analyzer = QuickFieldAnalyzer(filename)
    analyzer.run_comprehensive_analysis()

def compare_multiple_files(files: List[str]):
    """Сравнение нескольких файлов результатов"""
    print("🔍 СРАВНИТЕЛЬНЫЙ АНАЛИЗ ФАЙЛОВ")
    print("="*50)
    
    results = {}
    for file in files:
        if os.path.exists(file):
            print(f"\nАнализ файла: {file}")
            analyzer = QuickFieldAnalyzer(file)
            if analyzer.data is not None:
                # Быстрый анализ ключевых параметров
                potential = analyzer.data['potential']
                min_pot = np.min(potential)
                max_pot = np.max(potential)
                mean_pot = np.mean(potential)
                
                # Потенциал в нанопроводе (примерная область)
                X, Y, Z = analyzer.data['grid']
                nanowire_mask = (
                    (X >= -10) & (X <= 10) & 
                    (Y >= -4) & (Y <= 4) & 
                    (Z >= 1) & (Z <= 5)
                )
                nanowire_mean = np.mean(potential[nanowire_mask])
                
                results[file] = {
                    'min_potential': min_pot,
                    'max_potential': max_pot,
                    'mean_potential': mean_pot,
                    'nanowire_mean': nanowire_mean,
                    'well_depth': 10.0 - min_pot  # относительно затвора
                }
                
                print(f"  Минимальный потенциал: {min_pot:.3f} V")
                print(f"  Глубина ямы: {10.0 - min_pot:.3f} V")
                print(f"  Средний в нанопроводе: {nanowire_mean:.3f} V")
    
    # Визуализация сравнения
    if len(results) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        files_list = list(results.keys())
        well_depths = [results[f]['well_depth'] for f in files_list]
        nanowire_means = [results[f]['nanowire_mean'] for f in files_list]
        
        # График глубин ям
        axes[0].bar(files_list, well_depths, color='skyblue', alpha=0.7)
        axes[0].set_ylabel('Глубина потенциальной ямы (V)')
        axes[0].set_title('Сравнение глубины ям')
        axes[0].tick_params(axis='x', rotation=45)
        
        # График средних потенциалов
        axes[1].bar(files_list, nanowire_means, color='lightgreen', alpha=0.7)
        axes[1].set_ylabel('Средний потенциал в нанопроводе (V)')
        axes[1].set_title('Сравнение средних потенциалов')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()

# Пример использования
if __name__ == "__main__":
    # Быстрый анализ одного файла
    quick_analyze("electric_field_results.pkl")
    
    # Сравнение нескольких файлов (если есть)
    # files_to_compare = [
    #     "results_basic.pkl",
    #     "results_high_epsilon.pkl", 
    #     "results_different_geometry.pkl"
    # ]
    # compare_multiple_files(files_to_compare)