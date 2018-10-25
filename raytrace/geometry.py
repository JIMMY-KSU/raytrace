import numpy as np
from .vector import *
from .render import *


class Ground:
    
    def __init__(self, position, normal):
        self.normal = normal.unit()
        self.position = self.normal * position.dot(self.normal)
    
    def intersections(self, ray, invert = False):
        if invert:
            return Ground(self.position, self.normal * -1).intersections(ray)
        
        else:
            area = len(ray)
            
            directions_para = self.normal.dot(ray.v) * -1
            positions_para = self.normal.dot(ray.r) * -1
            
            mask = np.logical_and(
                directions_para > 0,
                (ray.r - self.position).dot(self.normal) > 0
            )
            
            directions_para_set = np.extract(mask, directions_para)
            positions_para_set = np.extract(mask, positions_para)
            
            distance_set = (self.position.norm() - positions_para_set) / directions_para_set
            normal_set = self.normal.repeat(len(distance_set))
            
            if self.normal.allEqual(V3(1, 0, 0)):
                udir = V3(0, 1, 0)
                vdir = V3(0, 0, 1)
            else:
                udir = V3(0, 1, 0).cross(self.normal).unit()
                vdir = self.normal.cross(udir)
            
            ray_set = ray.extract(mask)
            incident_set = ray_set.trace(distance_set) - self.position
            u_set = incident_set.dot(udir)
            v_set = incident_set.dot(vdir)
            
            collisions = CollisionResult(area)
            collisions.place(mask, incident_set, normal_set, u_set, v_set)
            
            return collisions
    
    def interior(self, point):
        return (point - self.position).dot(self.normal) < 0


class Sphere:
    
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius
        
    def intersections(self, ray, invert = False):
        area = len(ray)
        
        x = ray.r - self.center
        x_para = ray.v.scale(x.dot(ray.v))
        x_perp = x - x_para
        
        mask = np.logical_and(
            np.logical_or(
                x.normsq() > self.radius**2,
                invert
            ),
            np.logical_and(
                x_perp.normsq() <= self.radius**2,
                x.dot(ray.v) < 0
            )
        )
        
        x_para_set = x_para.extract(mask)
        x_perp_set = x_perp.extract(mask)
        ray_set = ray.extract(mask)
        
        y_para_orientation = 1 if invert else -1
        y_para_set = ray_set.v.scale(y_para_orientation * np.sqrt(self.radius ** 2 - x_perp_set.normsq()))
        
        distance_set = (x_para_set - y_para_set).norm()
        incident_set = ray_set.trace(distance_set)
        
        normal_set = (x_perp_set + y_para_set).unit()
        if invert:
            normal_set = normal_set * -1
        
        u_set = np.arctan2(normal_set.y, normal_set.x) / (2.0 * np.pi)
        v_set = np.arccos(normal_set.z) / np.pi
        
        collisions = CollisionResult(area)
        collisions.place(mask, incident_set, normal_set, u_set, v_set)
        
        return collisions
    
    def interior(self, point):
        x = point - self.center
        return x.normsq() < self.radius ** 2


class Union:
    
    def __init__(self, fst, snd):
        self.fst = fst
        self.snd = snd
    
    def intersections(self, ray, invert = False):
        collisions_fst = self.fst.intersections(ray, invert)
        collisions_snd = self.snd.intersections(ray, invert)
        collisions = collisions_fst
        collisions.takeNearer(collisions_snd)
        return collisions
    
    def interior(self, point):
        interior_fst = self.fst.interior(point)
        interior_snd = self.snd.interior(point)
        return np.logical_and(interior_fst, interior_snd)


class Difference:
    
    def __init__(self, positive, negative):
        self.positive = positive
        self.negative = negative
    
    def intersections(self, ray, invert = False):
        collisions_p = self.positive.intersections(ray, invert)
        mask_p = self.negative.interior(collisions_p.incd)
        collisions_p.incd.place(mask_p, V3(np.inf, np.inf, np.inf))
        
        collisions_n = self.negative.intersections(ray, not invert)
        mask_n = np.logical_not(self.positive.interior(collisions_n.incd))
        collisions_n.incd.place(mask_n, V3(np.inf, np.inf, np.inf))
        
        collisions = collisions_p
        
        mask = collisions_n.incd.normsq() < collisions_p.incd.normsq()
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
    
    def intersections(self, ray, invert = False):
        collisions_1 = self.fst.intersections(ray, invert)
        mask_1 = np.logical_not(self.snd.interior(collisions_1.incd))
        collisions_1.incd.place(mask_1, V3(np.inf, np.inf, np.inf))
        
        collisions_2 = self.snd.intersections(ray, invert)
        mask_2 = np.logical_not(self.fst.interior(collisions_2.incd))
        collisions_2.incd.place(mask_2, V3(np.inf, np.inf, np.inf))
        
        collisions = collisions_1
        
        mask = collisions_2.incd.normsq() < collisions_1.incd.normsq()
        collisions.copyfrom(mask, collisions_2)
        
        return collisions
    
    def interior(self, point):
        in_fst = self.fst.interior(point)
        in_snd = self.snd.interior(point)
        return np.logical_and(in_fst, in_snd)


class Transformation:
    
    def intersections(self, ray, invert = False):
        ray_p = ray.transform(self.inverse)
        collisions_p = self.obj.intersections(ray_p, invert)
        collisions = collisions_p.transform(self.transform)
        return collisions
    
    def interior(self, point):
        point_p = self.inverse.apply(point)
        return self.obj.interior(point_p)


class Translation(Transformation):
    
    def __init__(self, delta, obj):
        self.transform = TranslationHelper(delta)
        self.inverse = self.transform.inverse()
        self.obj = obj


class Scaling(Transformation):
    
    def __init__(self, factor, obj):
        self.transform = ScalingHelper(factor)
        self.inverse = self.transform.inverse()
        self.obj = obj


class Rotation(Transformation):
    
    def __init__(self, axisIndex, angle, obj):
        self.transform = RotationHelper(axisIndex, angle)
        self.inverse = self.transform.inverse()
        self.obj = obj
