import numpy as np
from .vector import *


class Sphere:
    
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius
        
    def intersections(self, rays, invert = False):
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
        
        y_para_orientation = 1 if invert else -1
        y_para_set = directions_set.scale(y_para_orientation * np.sqrt(self.radius ** 2 - x_perp_set.normsq()))
        
        distance = np.repeat(np.inf, len(mask))
        distance_set = (x_para_set - y_para_set).norm()
        np.place(distance, mask, distance_set)
        
        normal = V3(0.0, 0.0, 0.0).repeat(len(mask))
        normal_set = (x_perp_set + y_para_set).unit()
        normal.place(mask, normal_set)
        if invert:
            normal = normal * -1
        
        return (distance, normal)
    
    def interior(self, point):
        x = point - self.center
        return x.normsq() < self.radius ** 2


class Difference:
    
    def __init__(self, positive, negative):
        self.positive = positive
        self.negative = negative
    
    def intersections(self, rays, invert = False):
        (distance_p, normal_p) = self.positive.intersections(rays, invert)
        incident_p = rays[0] + rays[1] * distance_p
        mask_p = self.negative.interior(incident_p)
        np.place(distance_p, mask_p, np.inf)
        
        (distance_n, normal_n) = self.negative.intersections(rays, not invert)
        incident_n = rays[0] + rays[1] * distance_n
        mask_n = np.logical_not(self.positive.interior(incident_n))
        np.place(distance_n, mask_n, np.inf)
        
        distance = distance_p
        normal = normal_p
        
        mask = distance_n < distance_p
        np.place(distance, mask, np.extract(mask, distance_n))
        normal.place(mask, normal_n.extract(mask))
        
        return (distance, normal)
    
    def interior(self, point):
        return np.logical_and(
            self.positive.interior(point),
            np.logical_not(
                self.negative.interior(point)
            )
        )


class Intersection:
    
    def __init__(self, fst, snd):
        self.fst = fst
        self.snd = snd
    
    def intersections(self, rays, invert = False):
        (distance_1, normal_1) = self.fst.intersections(rays, invert)
        incident_1 = rays[0] + rays[1] * distance_1
        mask_1 = np.logical_not(self.snd.interior(incident_1))
        np.place(distance_1, mask_1, np.inf)
        
        (distance_2, normal_2) = self.snd.intersections(rays, invert)
        incident_2 = rays[0] + rays[1] * distance_2
        mask_2 = np.logical_not(self.fst.interior(incident_2))
        np.place(distance_2, mask_2, np.inf)
        
        distance = distance_1
        normal = normal_1
        
        mask = distance_2 < distance_1
        np.place(distance, mask, np.extract(mask, distance_2))
        normal.place(mask, normal_2.extract(mask))
        
        return (distance, normal)
    
    def interior(self, point):
        in_fst = self.fst.interior(point)
        in_snd = self.snd.interior(point)
        return np.logical_and(in_fst, in_snd)