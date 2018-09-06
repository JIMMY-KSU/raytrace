import numpy as np

class Sphere:
    
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius
        
    def intersections(self, rays):
        positions =  rays[0]
        directions = rays[1]
        
        x = positions - self.center
        x_para = directions.scale(x.dot(directions))
        x_perp = x - x_para
        y_para = directions.scale(-1 * np.sqrt(self.radius ** 2 - x_perp.normsq()))
        
        distance = (x_para - y_para).norm()
        miss = np.logical_or(
            x.normsq() <= self.radius**2,
            np.logical_or(
                x_perp.normsq() > 1,
                x.dot(directions) >= 0
            )
        )
        np.place(distance, miss, np.inf)
        
        normal = (x_perp + y_para).unit()
        
        return (distance, normal)