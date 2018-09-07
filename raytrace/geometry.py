import numpy as np
from .vector import *

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
        
        mask = np.logical_and(
            x.normsq() > self.radius**2,
            np.logical_and(
                x_perp.normsq() <= self.radius**2,
                x.dot(directions) < 0
            )
        )
        
        x_para_set = x_para.extract(mask)
        x_perp_set = x_perp.extract(mask)
        directions_set = directions.extract(mask)
        
        y_para_set = directions_set.scale(-1 * np.sqrt(self.radius ** 2 - x_perp_set.normsq()))
        
        distance = np.repeat(np.inf, len(mask))
        distance_set = (x_para_set - y_para_set).norm()
        np.place(distance, mask, distance_set)
        
        normal = V3(0.0, 0.0, 0.0).repeat(len(mask))
        normal_set = (x_perp_set + y_para_set).unit()
        normal.place(mask, normal_set)
        
        return (distance, normal)