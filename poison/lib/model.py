from typing import List, Tuple
from poison.lib.entity import Point, Volume

import numpy as np
import math

class Box(Volume):    
    def __init__(self, x: float, y: float, z: float, dx: float, dy: float, dz: float, mesh_size: float = None):
        self.x = x
        self.y = y
        self.z = z
        self.dx = dx
        self.dy = dy
        self.dz = dz
    
    def GetCenter(self) -> Point:
        return Point(self.x + self.dx/2, self.y + self.dy/2, self.z + self.dz/2)
    
    def GetOrigin(self) -> Point:
        return Point(self.x, self.y, self.z)
    
    def GetDimensions(self) -> Tuple[float, float, float]:
        return (self.dx, self.dy, self.dz)

    def GetCoords(self) -> Tuple[float, float, float, float, float, float]:
        return (self.x, self.y, self.z, self.dx, self.dy, self.dz)
    
    def GetBounds(self) -> List[float]:
        return [self.x, self.x + self.dx, self.y, self.y + self.dy, self.z, self.z + self.dz]
    
    def __repr__(self):
        return f"Box(x={self.x}, y={self.y}, z={self.z}, dx={self.dx}, dy={self.dy}, dz={self.dz})"

class Envelope:    
    def __init__(self, bridge_center_x: float, bridge_center_y: float, bridge_center_z: float,
                 grip_width: float, grip_height: float, envelope_gap: float, 
                 envelope_thickness: float, envelope_length: float,
                 mesh_size: float):
        """
        Args:
            bridge_center_x, y, z: центр узкой части наномостика
            grip_width: ширина узкой части наномостика
            grip_height: высота узкой части наномостика  
            envelope_gap: зазор между электродом и наномостиком
            envelope_thickness: толщина слоя
            envelope_length: длина слоя
            mesh_size: размер сетки
        """
        self.bridge_center_x = bridge_center_x
        self.bridge_center_y = bridge_center_y
        self.bridge_center_z = bridge_center_z
        self.grip_width = grip_width
        self.grip_height = grip_height
        self.envelope_gap = envelope_gap
        self.envelope_length = envelope_length
        self.envelope_thickness = envelope_thickness
        self.mesh_size = mesh_size
        
        self.volumes = []
        
    def GetVolumes(self):
        return self.volumes

    def create_c_shaped(self, angle=180):
        width = self.grip_width + 2 * self.envelope_thickness
        height = self.grip_height + self.envelope_thickness
        angle_min = math.atan2(height, width / 2)
        angle_max = math.pi - angle_min
        angle = math.radians(angle)

        if angle < angle_min:
            z_height = width * math.tan(angle) / 2

            self.volumes.append(Box(
                self.bridge_center_x - self.envelope_length/2,
                self.bridge_center_y - width/2,
                self.bridge_center_z,
                self.envelope_length,
                self.envelope_thickness,
                z_height,
                self.mesh_size
            ))
        else:
            self.volumes.append(Box(
                self.bridge_center_x - self.envelope_length/2,
                self.bridge_center_y - width/2,
                self.bridge_center_z,
                self.envelope_length,
                self.envelope_thickness,
                height,
                self.mesh_size
            ))

        if angle > angle_min and angle < angle_max:
            length = (math.sqrt((width/2)**2 + height**2) * math.sin(angle - angle_min)) / math.sin(angle)
            self.volumes.append(Box(
                self.bridge_center_x - self.envelope_length/2,
                self.bridge_center_y - self.grip_width/2,
                self.bridge_center_z + height - self.envelope_thickness,
                self.envelope_length,
                length,
                self.envelope_thickness,
                self.mesh_size
            ))
        elif angle >= angle_max:
            self.volumes.append(Box(
                self.bridge_center_x - self.envelope_length/2,
                self.bridge_center_y - self.grip_width/2,
                self.bridge_center_z + height - self.envelope_thickness,
                self.envelope_length,
                self.grip_width,
                self.envelope_thickness,
                self.mesh_size
            ))

        if angle > angle_max and angle < math.pi:
            self.volumes.append(Box(
                self.bridge_center_x - self.envelope_length/2,
                self.bridge_center_y + width/2 - self.envelope_thickness,
                self.bridge_center_z + height,
                self.envelope_length,
                self.envelope_thickness,
                -(height + width * math.tan(angle) / 2),
                self.mesh_size
            ))
        elif angle >= math.pi:
            self.volumes.append(Box(
                self.bridge_center_x - self.envelope_length/2,
                self.bridge_center_y + width/2 - self.envelope_thickness,
                self.bridge_center_z,
                self.envelope_length,
                self.envelope_thickness,
                height,
                self.mesh_size
            ))


class NanoBridge:    
    def __init__(self, grip_length=10, grip_width=5, grip_height=5,
                 end_length=5, end_width=10, mesh_size=0.2):
        self.grip_length = grip_length
        self.grip_width = grip_width
        self.grip_height = grip_height
        self.end_length = end_length
        self.end_width = end_width
        self.mesh_size = mesh_size
        
        self.boxes = []
        self.oxide_boxes = []
    
    def create_nano_bridge_geometry(self):
        gl, gw, gh, el, ew, ms = self.grip_length, self.grip_width, self.grip_height, self.end_length, self.end_width, self.mesh_size

        left_box = Box(
            -gl/2 - el, -ew/2, 0,
            el, ew, gh,
            ms
        )
        
        center_box = Box(
            -gl/2, -gw/2, 0,
            gl, gw, gh,
            ms
        )
        
        right_box = Box(
            gl/2, -ew/2, 0,
            el, ew, gh,
            ms
        )
        
        self.boxes = [left_box, center_box, right_box]
        return self.boxes

    def create_oxide(self, oxide_thickness=0.3):
        (left, center, right) = self.get_boxes()
        leftCenter = left.GetCenter().GetCoords()
        centerCenter = center.GetCenter().GetCoords()
        rightCenter = right.GetCenter().GetCoords()
        
        
        ox1 = Envelope(
            bridge_center_x=centerCenter[0], bridge_center_y=centerCenter[1], bridge_center_z=0,
            grip_width=self.grip_width, grip_height=self.grip_height,
            envelope_gap=0, envelope_thickness=oxide_thickness, 
            envelope_length=self.grip_length,
            mesh_size=0.4
        )
        ox1.create_c_shaped()  
        
        ox2 = Envelope(
            bridge_center_x=leftCenter[0], bridge_center_y=leftCenter[1], bridge_center_z=0,
            grip_width=self.end_width, grip_height=self.grip_height,
            envelope_gap=0, envelope_thickness=oxide_thickness,
            envelope_length=self.end_length,
            mesh_size=0.4
        )
        ox2.create_c_shaped()  
        
        ox3 = Envelope(
            bridge_center_x=rightCenter[0], bridge_center_y=rightCenter[1], bridge_center_z=0,
            grip_width=self.end_width, grip_height=self.grip_height,
            envelope_gap=0, envelope_thickness=oxide_thickness,
            envelope_length=self.end_length,
            mesh_size=0.4
        )
        ox3.create_c_shaped()

        # Добавляем все оксидные боксы в один список
        self.oxide_boxes = ox1.GetVolumes() + ox2.GetVolumes() + ox3.GetVolumes()
        print(f"Создано оксидных боксов: {len(self.oxide_boxes)}")
    
    def get_boxes(self) -> List[Box]:
        return self.boxes

    def get_oxide_boxes(self) -> List[Box]:
        return self.oxide_boxes
    
    def get_total_bounds(self) -> List[float]:
        """Возвращает общие границы наномостика"""
        all_boxes = self.boxes + self.oxide_boxes
        all_bounds = [box.GetBounds() for box in all_boxes]
        
        bounds_array = np.array(all_bounds)
        return [
            bounds_array[:, 0].min(),  # x_min
            bounds_array[:, 1].max(),  # x_max
            bounds_array[:, 2].min(),  # y_min
            bounds_array[:, 3].max(),  # y_max
            bounds_array[:, 4].min(),  # z_min
            bounds_array[:, 5].max()   # z_max
        ]

class Substrate:    
    def __init__(self, thickness=5, padding=10, mesh_size=1.0):
        self.thickness = thickness
        self.padding = padding
        self.mesh_size = mesh_size
        self.box = None
    
    def create_substrate(self, nano_bounds):
        x_min, x_max, y_min, y_max, z_min, z_max = nano_bounds
        
        # Подложка располагается под наномостиком
        substrate_z = z_min - self.thickness
        
        # Подложка расширяется за пределы наномостика
        substrate_x_min = x_min - self.padding
        substrate_x_max = x_max + self.padding
        substrate_y_min = y_min - self.padding
        substrate_y_max = y_max + self.padding
        
        substrate_dx = substrate_x_max - substrate_x_min
        substrate_dy = substrate_y_max - substrate_y_min
        
        self.box = Box(
            substrate_x_min, substrate_y_min, substrate_z,
            substrate_dx, substrate_dy, self.thickness,
            self.mesh_size
        )
        
        return self.box

class AirEnvironment:    
    def __init__(self, padding=15, height_above=20, height_below=10, mesh_size=1.5):
        self.padding = padding
        self.height_above = height_above
        self.height_below = height_below
        self.mesh_size = mesh_size
        self.box = None
    
    def create_air_environment(self, nano_bounds):
        x_min, x_max, y_min, y_max, z_min, z_max = nano_bounds
        
        # Окружающая среда охватывает все элементы
        air_x_min = x_min - self.padding
        air_x_max = x_max + self.padding
        air_y_min = y_min - self.padding
        air_y_max = y_max + self.padding
        air_z_min = z_min - self.height_below
        air_z_max = z_max + self.height_above
        
        air_dx = air_x_max - air_x_min
        air_dy = air_y_max - air_y_min
        air_dz = air_z_max - air_z_min
        
        self.box = Box(
            air_x_min, air_y_min, air_z_min,
            air_dx, air_dy, air_dz,
            self.mesh_size
        )
        
        return self.box

class GateElectrode:    
    def __init__(self, center_point: Point, width=15, height=8, length=5, thickness=2, mesh_size=0.6, coverage_angle=180, gap_distance=0):
        self.center = center_point
        self.width = width
        self.height = height
        self.length = length
        self.thickness = thickness
        self.mesh_size = mesh_size
        self.boxes = []
        self.coverage_angle = coverage_angle
        self.gap_distance = gap_distance
    
    def create_gate_electrode(self):
        gate = Envelope(
            bridge_center_x=self.center.x,
            bridge_center_y=self.center.y, 
            bridge_center_z=self.center.z - self.height / 2,
            grip_width=self.width,
            grip_height=self.height,
            envelope_gap=self.gap_distance,
            envelope_thickness=self.thickness,
            envelope_length=self.length,
            mesh_size=self.mesh_size
        )
        
        gate.create_c_shaped(self.coverage_angle)
        self.boxes = gate.GetVolumes()

        mainPart = gate.GetVolumes()
        
        leftHand = Box(
            mainPart[0].GetOrigin().x, 
            mainPart[0].GetOrigin().y, 
            mainPart[0].GetOrigin().z,
            mainPart[0].GetDimensions()[0], 
            -self.thickness * 5, 
            self.thickness,
            self.mesh_size
        )
        self.boxes.append(leftHand)

        if self.coverage_angle >= 180:
            rightHand = Box(
                mainPart[2].GetOrigin().x, 
                mainPart[2].GetOrigin().y + self.thickness, 
                mainPart[2].GetOrigin().z,
                mainPart[2].GetDimensions()[0], 
                self.thickness * 5, 
                self.thickness,
                self.mesh_size
            )

            self.boxes.append(rightHand)
        
        return self.boxes

class CompleteNanoSystem:
    def __init__(self, config):
        self.config = config
        self.nano_bridge = None
        self.substrate = None
        self.air_environment = None
        self.gate_electrodes = []  # Теперь список электродов
        self.electrode_potentials = {}  # Словарь: имя электрода -> потенциал
        self.all_components = []
    
    def create_complete_system(self):
        nb_config = self.config['nanobridge']
        sub_config = self.config['substrate']
        air_config = self.config['air_environment']
        
        # Используем новый массив electrodes вместо старого electrode
        electrodes_config = self.config.get('electrodes', [])
        if not electrodes_config:
            # Fallback на старый формат для обратной совместимости
            if 'electrode' in self.config:
                electrodes_config = [self.config['electrode']]
                print("⚠️ Используется устаревший формат 'electrode'. Рекомендуется перейти на 'electrodes'")

        self.nano_bridge = NanoBridge(
            grip_length=nb_config['grip_length'], grip_width=nb_config['grip_width'], grip_height=nb_config['grip_height'],
            end_length=nb_config['end_length'], end_width=nb_config['end_width'], mesh_size=nb_config['mesh_size']
        )
        self.nano_bridge.create_nano_bridge_geometry()
        self.nano_bridge.create_oxide(nb_config["oxide_thickness"])
        
        nano_bounds = self.nano_bridge.get_total_bounds()
        self.substrate = Substrate(thickness=sub_config['thickness'], padding=sub_config['padding'], mesh_size=sub_config['mesh_size'])
        substrate_box = self.substrate.create_substrate(nano_bounds)
        
        all_bounds_so_far = nano_bounds + substrate_box.GetBounds()
        bounds_array = np.array([all_bounds_so_far[:6], all_bounds_so_far[6:]])
        total_bounds = [
            bounds_array[:, 0].min(), bounds_array[:, 1].max(),
            bounds_array[:, 2].min(), bounds_array[:, 3].max(),
            bounds_array[:, 4].min(), bounds_array[:, 5].max()
        ]
        
        self.air_environment = AirEnvironment(padding=air_config['padding'], height_above=air_config['height_above'],
                                              height_below=air_config['height_below'], mesh_size=air_config['mesh_size'])
        air_box = self.air_environment.create_air_environment(total_bounds)
        
        (_, center_box, _) = self.nano_bridge.get_boxes()
        center_point = center_box.GetCenter()
        
        oxide_thickness = nb_config.get('oxide_thickness', 0)
        
        # Создаем все электроды из массива
        all_gate_boxes = []
        for el_config in electrodes_config:
            gap = el_config.get('gap_distance', 0)
            electrode_width = nb_config['grip_width'] + 2 * (oxide_thickness + gap)
            electrode_height = nb_config['grip_height'] + oxide_thickness + gap
            electrode_center_z = (nb_config['grip_height'] + oxide_thickness + gap) / 2
            
            # Используем center_x из конфига, если указан, иначе центр наномостика
            center_x = el_config.get('center_x', center_point.x)
            electrode_center = Point(center_x, center_point.y, electrode_center_z)

            gate_electrode = GateElectrode(
                center_point=electrode_center, width=electrode_width, height=electrode_height,
                length=el_config['length'], thickness=el_config['thickness'], mesh_size=el_config['mesh_size'],
                coverage_angle=el_config['coverage_angle'], gap_distance=gap
            )
            gate_boxes = gate_electrode.create_gate_electrode()
            all_gate_boxes.extend(gate_boxes)
            
            self.gate_electrodes.append(gate_electrode)
            
            # Сохраняем потенциал для каждого электрода
            electrode_name = el_config.get('name', f'electrode_{len(self.gate_electrodes)}')
            self.electrode_potentials[electrode_name] = {
                'potential': el_config.get('potential', 0.0),
                'boxes': gate_boxes
            }
            
            print(f"✓ Создан электрод '{electrode_name}' в x={center_x:.1f} нм, потенциал={el_config.get('potential', 0.0):.2f} В")
        
        self.all_components = {
            'nano_bridge': self.nano_bridge.get_boxes(),
            'oxide': self.nano_bridge.get_oxide_boxes(),
            'substrate': [substrate_box],
            'air': [air_box],
            'gate': all_gate_boxes
        }
        
        return self.all_components