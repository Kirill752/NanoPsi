from typing import List, Tuple

class Point:
    def __init__(self, x: float, y: float, z: float = 0):
        self.x = x
        self.y = y
        self.z = z
    
    def GetCoords(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)
    
    def __repr__(self):
        return f"Point({self.x}, {self.y}, {self.z})"

class Line:
    def __init__(self, p1: Point, p2: Point):
        self.p1 = p1
        self.p2 = p2

    def GetPoints(self) -> Tuple[Point, Point]:
        return (self.p1, self.p2)
    
    def __repr__(self):
        return f"Line({self.p1.id}->{self.p2.id}, id={self.id})"


class Surface:
    def __init__(self, lines: List[Line], mesh_size: float = None):
        self.lines = lines
    
    def GetLines(self) -> List[Line]:
        return self.lines
    
    def __repr__(self):
        return f"Surface(lines={len(self.lines)}, id={self.id})"

class Volume:
    def __init__(self, id = None):
        self.mesh_size = None
        self.id = id
    
    def GetId(self) -> int:
        return self.id
    
    def __repr__(self):
        return f"Volume(id={self.id})"