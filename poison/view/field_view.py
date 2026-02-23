from poison.lib.model import CompleteNanoSystem
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

class NanoSystemVisualizer:    
    def __init__(self, nano_system: CompleteNanoSystem):
        self.nano_system = nano_system
        self.pyvista_objects = []
    
    def convert_to_pyvista(self):
        self.pyvista_objects = []
        
        components = self.nano_system.all_components
        colors = {
            'nano_bridge': ['blue', 'blue' , 'blue'],
            'oxide': ['darkgreen'] * len(components['oxide']),
            'substrate': ['gray'],
            'air': ['lightblue'],
            'gate': ['purple'] * len(components['gate'])
        }
        
        names = {
            'nano_bridge': ['Nano Bridge'] * len(components['nano_bridge']),
            'oxide': ['Oxide'] * len(components['oxide']),
            'substrate': ['Substrate'],
            'air': ['Air'],
            'gate': ['Gate'] * len(components['gate'])
        }
        
        component_idx = 0
        for comp_type, boxes in components.items():
            for i, box in enumerate(boxes):
                x, y, z, dx, dy, dz = box.GetCoords()
                
                x_min = min(x, x + dx)
                x_max = max(x, x + dx)
                y_min = min(y, y + dy)
                y_max = max(y, y + dy)
                z_min = min(z, z + dz)
                z_max = max(z, z + dz)
                
                pv_box = pv.Box(bounds=[
                    x_min, x_max,
                    y_min, y_max,  
                    z_min, z_max
                ])
                
                n_points = pv_box.n_points
                pv_box.point_data["component"] = np.full(n_points, component_idx)
                
                color = colors[comp_type][i] if comp_type in colors else 'white'
                name = names[comp_type][i] if comp_type in names else f'{comp_type}_{i}'
                
                self.pyvista_objects.append({
                    'mesh': pv_box,
                    'name': name,
                    'color': color,
                    'type': comp_type
                })
                component_idx += 1
        
        print(f"Всего объектов для визуализации: {len(self.pyvista_objects)}")
        return self.pyvista_objects
    
    def visualize_complete_system(self):
        """Визуализация полной системы"""
        if not self.pyvista_objects:
            self.convert_to_pyvista()
        
        plotter = pv.Plotter()
        
        added_to_legend = {
            'nano_bridge': False,
            'oxide': False, 
            'substrate': False,
            'air': False,
            'gate': False
        }
        
        for obj in self.pyvista_objects:
            opacity = 0.3 if obj['type'] == 'air' else 0.8 if obj['type'] == 'oxide' else 1.0
            show_edges = obj['type'] != 'air'
            
            if obj['type'] in added_to_legend and not added_to_legend[obj['type']]:
                label = obj['name']
                added_to_legend[obj['type']] = True
            else:
                label = None
            
            plotter.add_mesh(obj['mesh'], 
                           color=obj['color'], 
                           opacity=opacity,
                           edge_color='black' if show_edges else None,
                           line_width=1,
                           show_edges=show_edges,
                           label=label)
        
        plotter.add_legend()
        plotter.show_axes()
        plotter.show_grid(
            xtitle="X (nm)",
            ytitle="Y (nm)",
            ztitle="Z (nm)",
            color='black',
            font_size=10,
            show_xaxis=True,
            show_yaxis=True,
            show_zaxis=True
        )
        plotter.show()
