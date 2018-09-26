import numpy as np
from .vector import *
from .render import *


class Ground:
    
    def __init__(self, position, normal):
        self.normal = normal.unit()
        self.position = self.normal * position.dot(self.normal)
    
    def intersections(self, rays, invert = False):
        positions = rays[0]
        directions = rays[1]
        
        area = len(positions)
        
        if invert:
            return Ground(self.position, self.normal * -1).intersections(rays)
        
        else:
            directions_para = self.normal.dot(directions) * -1
            positions_para = self.normal.dot(positions) * -1
            
            mask = np.logical_and(
                directions_para > 0,
                (positions - self.position).dot(self.normal) > 0
            )
            
            directions_para_set = np.extract(mask, directions_para)
            positions_para_set = np.extract(mask, positions_para)
            
            distance_set = (self.position.norm() - positions_para_set) / directions_para_set
            normal_set = self.normal.repeat(len(distance_set))
            
            collisions = CollisionResult(area)
            collisions.place(mask, distance_set, normal_set)
            
            return collisions
    
    def interior(self, point):
        return (point - self.position).dot(self.normal) < 0


class Sphere:
    
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius
        
    def intersections(self, rays, invert = False):
        positions =  rays[0]
        directions = rays[1]
        
        area = len(positions)
        
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
        
        distance_set = (x_para_set - y_para_set).norm()
        
        normal_set = (x_perp_set + y_para_set).unit()
        if invert:
            normal_set = normal_set * -1
        
        collisions = CollisionResult(area)
        collisions.place(mask, distance_set, normal_set)
        
        return collisions
    
    def interior(self, point):
        x = point - self.center
        return x.normsq() < self.radius ** 2


class Difference:
    
    def __init__(self, positive, negative):
        self.positive = positive
        self.negative = negative
    
    def intersections(self, rays, invert = False):
        collisions_p = self.positive.intersections(rays, invert)
        incident_p = rays[0] + rays[1] * collisions_p.dist
        mask_p = self.negative.interior(incident_p)
        np.place(collisions_p.dist, mask_p, np.inf)
        
        collisions_n = self.negative.intersections(rays, not invert)
        incident_n = rays[0] + rays[1] * collisions_n.dist
        mask_n = np.logical_not(self.positive.interior(incident_n))
        np.place(collisions_n.dist, mask_n, np.inf)
        
        collisions = collisions_p
        
        mask = collisions_n.dist < collisions_p.dist
        collisions.copyfrom(mask, collisions_n)
        
        return collisions
    
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
        collisions_1 = self.fst.intersections(rays, invert)
        incident_1 = rays[0] + rays[1] * collisions_1.dist
        mask_1 = np.logical_not(self.snd.interior(incident_1))
        np.place(collisions_1.dist, mask_1, np.inf)
        
        collisions_2 = self.snd.intersections(rays, invert)
        incident_2 = rays[0] + rays[1] * collisions_2.dist
        mask_2 = np.logical_not(self.fst.interior(incident_2))
        np.place(collisions_2.dist, mask_2, np.inf)
        
        collisions = collisions_1
        
        mask = collisions_2.dist < collisions_1.dist
        collisions.copyfrom(mask, collisions_2)
        
        return collisions
    
    def interior(self, point):
        in_fst = self.fst.interior(point)
        in_snd = self.snd.interior(point)
        return np.logical_and(in_fst, in_snd)