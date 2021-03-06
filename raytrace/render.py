import numpy as np

from .vector import *
from .camera import *
from .geometry import *
from .material import *
from .transform import *


class CollisionResult:
    
    def __init__(self, area):
        self.area = area
        self.mathash = np.repeat([0], area)
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
        np.copyto(self.mathash, other.mathash, where = mask)
        self.incd.copyfrom(other.incd, where = mask)
        self.norm.copyfrom(other.norm, where = mask)
        np.copyto(self.u, other.u, where = mask)
        np.copyto(self.v, other.v, where = mask)
    
    def setMatHash(self, mathash):
        self.mathash = np.repeat([mathash], self.area)
    
    def takeNearer(self, other):
        mask = other.incd.normsq() < self.incd.normsq()
        self.copyfrom(mask, other)
    
    def transform(self, transform):
        result = CollisionResult(self.area)
        result.mathash = self.mathash
        result.incd = transform.applyToDifference(self.incd)
        result.norm = transform.applyToNormal(self.norm)
        result.u = self.u
        result.v = self.v
        return result
    
    def extract(self, mask):
        result = CollisionResult(np.sum(mask))
        result.mathash = np.extract(mask, self.mathash)
        result.incd = self.incd.extract(mask)
        result.norm = self.norm.extract(mask)
        result.u = np.extract(mask, self.u)
        result.v = np.extract(mask, self.v)
        return result


def collide(area, ray, scene):
    nearest_collisions = CollisionResult(area)
    for i, obj in enumerate(scene['objects']):
        obj_collisions = obj.intersections(ray)
        nearest_collisions.takeNearer(obj_collisions)
    return nearest_collisions


def render(camera, scene, bounce = 4):
    area = camera.area()
    ray = camera.rays()
    
    materials = {hash(key): scene['materials'][key] for key in scene['materials']}
    
    all_collisions = collide(area, ray, scene)
    collision_mask = (all_collisions.incd.x != np.inf)
    sub_area = np.sum(collision_mask)
    collisions = all_collisions.extract(collision_mask)
    
    # lighting and texturing
    
    lighting = V3(0, 0, 0).repeat(sub_area)
    for light in scene['lighting']:
        lighting += light.illuminate(scene, collisions)
    
    matte_component = V3(0, 0, 0).repeat(sub_area)
    reflective_component = V3(0, 0, 0).repeat(sub_area)
    
    frac_reflective = np.repeat([0.0], sub_area)
    for material_hash in materials:
        mask = (collisions.mathash == material_hash)
        material = materials[material_hash]
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
    
    sub_raster = V3(0, 0, 0).repeat(np.sum(collision_mask))
    sub_raster += matte_component * lighting * (1.0 - frac_reflective)
    sub_raster += reflective_component * frac_reflective
    
    raster = V3(0, 0, 0).repeat(area)
    raster.place(collision_mask, sub_raster)
    
    return raster.clip(0, 1)