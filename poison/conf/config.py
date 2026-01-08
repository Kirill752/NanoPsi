import os
import json
import yaml

class ConfigManager:
    """Менеджер конфигурации для загрузки параметров из файлов"""
    
    @staticmethod
    def load_config(config_file):
        """Загружает конфигурацию из JSON или YAML файла"""
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Конфигурационный файл {config_file} не найден!")
        
        with open(config_file, 'r') as f:
            if config_file.endswith('.json'):
                config = json.load(f)
            elif config_file.endswith('.yaml') or config_file.endswith('.yml'):
                config = yaml.safe_load(f)
            else:
                raise ValueError("Поддерживаются только JSON и YAML файлы")
        
        print(f"Конфигурация загружена из {config_file}")
        return config
    
    @staticmethod
    def save_config(config, config_file):
        """Сохраняет конфигурацию в файл"""
        with open(config_file, 'w') as f:
            if config_file.endswith('.json'):
                json.dump(config, f, indent=4)
            elif config_file.endswith('.yaml') or config_file.endswith('.yml'):
                yaml.dump(config, f, default_flow_style=False)
        
        print(f"Конфигурация сохранена в {config_file}")