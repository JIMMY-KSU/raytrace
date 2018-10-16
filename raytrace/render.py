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
        self.dist = np.repeat([np.inf], area)
        self.norm = V3(0, 0, 0).repeat(area)
        self.u = np.repeat([0.0], area)
        self.v = np.repeat([0.0], area)
        
    def place(self, mask, dist, norm, u = np.array([0]), v = np.array([0])):
        np.place(self.dist, mask, dist)
        self.norm.place(mask, norm)
        np.place(self.u, mask, u)
        np.place(self.v, mask, v)
        
    def copyfrom(self, mask, other):
        np.copyto(self.mat_index, other.mat_index, where = mask)
        np.copyto(self.dist, other.dist, where = mask)
        self.norm.copyfrom(other.norm, where = mask)
        np.copyto(self.u, other.u, where = mask)
        np.copyto(self.v, other.v, where = mask)
    
    def setMatIndex(self, mat_index):
        self.mat_index = np.repeat([mat_index], self.area)
    
    def takeNearer(self, other):
        mask = other.dist < self.dist
        self.copyfrom(mask, other)


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
    
    nearest_collisions = collide(area, ray, scene)
    
    # lighting and texturing
    
    raster = V3(0, 0, 0).repeat(area)
    
    directional_lighting = scene['lighting']['directional'].unit()
    shadow_ray = Ray(
        ray.trace(nearest_collisions.dist),
        (directional_lighting * -1).repeat(area)
    )
    shadow_collisions = collide(area, shadow_ray, scene)
    shadow_mask = (shadow_collisions.dist == np.inf)
    directional_component =  shadow_mask * np.clip(directional_lighting.dot(nearest_collisions.norm) * -1, 0, 1)
    
    ambient_lighting = scene['lighting']['ambient']
    lighting = (directional_component * (1 - ambient_lighting) + ambient_lighting)
    
    matte_component = V3(0.0, 0.0, 0.0).repeat(area)
    reflective_component = V3(0.0, 0.0, 0.0).repeat(area)
    
    reflectivity = np.repeat([0.0], area)
    
    for i, material_name in enumerate(material_list):
        mask = (nearest_collisions.mat_index == i)
        material = scene['materials'][material_name]
        u = np.extract(mask, nearest_collisions.u)
        v = np.extract(mask, nearest_collisions.v)
        matte_component.place(mask, material.getColor(u, v))
        np.place(reflectivity, mask, material.getReflectivity(u, v))
    
    reflective_mask = (reflectivity > 0.0)
    if bounce > 0 and reflective_mask.any():
        position_set = ray.trace(nearest_collisions.dist).extract(reflective_mask)
        incident_set = ray.v.extract(reflective_mask)
        normal_set = nearest_collisions.norm.extract(reflective_mask)
        reflected_set = incident_set - normal_set.unit() * incident_set.dot(normal_set) * 2
        reflective_camera = CameraPrecomputed(Ray(position_set, reflected_set))
        reflective_component_set = render(reflective_camera, scene, bounce - 1)
        reflective_component.place(reflective_mask, reflective_component_set)
    
    raster += matte_component * lighting * (1.0 - reflectivity)
    raster += reflective_component * reflectivity
    
    return raster