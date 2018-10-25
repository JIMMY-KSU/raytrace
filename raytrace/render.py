import numpy as np

from .vector import *
from .camera import *
from .geometry import *
from .material import *
from .transform import *


class CollisionResult:
    
    def __init__(self, area):
        self.area = area
        self.mat_index = np.repeat([-1], area)
        self.incd = V3(np.inf, np.inf, np.inf).repeat(area)
        self.norm = V3(1, 0, 0).repeat(area)
        self.u = np.repeat([0.0], area)
        self.v = np.repeat([0.0], area)
        
    def place(self, mask, incd, norm, u = np.array([0]), v = np.array([0])):
        self.incd.place(mask, incd)
        self.norm.place(mask, norm)
        np.place(self.u, mask, u)
        np.place(self.v, mask, v)
        
    def copyfrom(self, mask, other):
        np.copyto(self.mat_index, other.mat_index, where = mask)
        self.incd.copyfrom(other.incd, where = mask)
        self.norm.copyfrom(other.norm, where = mask)
        np.copyto(self.u, other.u, where = mask)
        np.copyto(self.v, other.v, where = mask)
    
    def setMatIndex(self, mat_index):
        self.mat_index = np.repeat([mat_index], self.area)
    
    def takeNearer(self, other):
        mask = other.incd.normsq() < self.incd.normsq()
        self.copyfrom(mask, other)
    
    def transform(self, transform):
        result = CollisionResult(self.area)
        result.mat_index = self.mat_index
        result.incd = transform.applyToDifference(self.incd)
        result.norm = transform.applyToNormal(self.norm)
        result.u = self.u
        result.v = self.v
        return result


def collide(area, ray, scene):
    nearest_collisions = CollisionResult(area)
    
    material_list = list(scene['materials'])
    material_indeces = {m: i for i, m in enumerate(material_list)}
    
    # compute nearest intersections
    
    for i, obj in enumerate(scene['objects']):
        obj_collisions = obj['geometry'].intersections(ray)
        obj_collisions.setMatIndex(material_indeces[obj['material']])
        nearest_collisions.takeNearer(obj_collisions)
    
    return nearest_collisions


def render(camera, scene, bounce = 4):
    area = camera.area()
    ray = camera.rays()
    
    material_list = list(scene['materials'])
    material_indeces = {m: i for i, m in enumerate(material_list)}
    
    collisions = collide(area, ray, scene)
    
    # lighting and texturing
    
    lighting = V3(0, 0, 0).repeat(area)
    for light in scene['lighting']:
        lighting += light.illuminate(scene, collisions)
    
    matte_component = V3(0, 0, 0).repeat(area)
    reflective_component = V3(0, 0, 0).repeat(area)
    
    frac_reflective = np.repeat([0.0], area)
    for i, material_name in enumerate(material_list):
        mask = (collisions.mat_index == i)
        material = scene['materials'][material_name]
        u = np.extract(mask, collisions.u)
        v = np.extract(mask, collisions.v)
        matte_component.place(mask, material.getColor(u, v))
        np.place(frac_reflective, mask, material.getReflectivity(u, v))
    
    reflective_mask = (frac_reflective > 0.0)
    if bounce > 0 and reflective_mask.any():
        position_set = collisions.incd.extract(reflective_mask)
        incident_set = ray.v.extract(reflective_mask)
        normal_set = collisions.norm.extract(reflective_mask)
        reflected_set = incident_set - normal_set.unit() * incident_set.dot(normal_set) * 2
        reflective_camera = CameraPrecomputed(Ray(position_set, reflected_set))
        reflective_component_set = render(reflective_camera, scene, bounce - 1)
        reflective_component.place(reflective_mask, reflective_component_set)
    
    raster = V3(0, 0, 0).repeat(area)
    raster += matte_component * lighting * (1.0 - frac_reflective)
    raster += reflective_component * frac_reflective
    
    return raster.clip(0, 1)