import numpy as np

from .vector import *
from .camera import *
from .geometry import *
from .material import *

class CollisionResult:
    def __init__(self, area):
        self.dist = np.repeat([np.inf], area)
        self.norm = V3(0, 0, 0).repeat(area)
    def place(self, mask, dist, norm):
        np.place(self.dist, mask, dist)
        self.norm.place(mask, norm)
    def copyfrom(self, mask, other):
        np.copyto(self.dist, other.dist, where = mask)
        self.norm.copyfrom(other.norm, where = mask)

def render(camera, scene, bounce = 4):
    area = camera.area()
    ray = camera.rays()
    
    nearest_material_index = np.repeat([-1], area)
    distance = np.repeat([np.inf], area)
    normal = V3(
        np.repeat([0.0], area),
        np.repeat([0.0], area),
        np.repeat([0.0], area)
    )
    raster = V3(
        np.repeat([0.0], area),
        np.repeat([0.0], area),
        np.repeat([0.0], area)
    )
    
    material_list = list(scene['materials'])
    material_indeces = {m: i for i, m in enumerate(material_list)}
    
    # compute nearest intersections
    
    for i, obj in enumerate(scene['objects']):
        material_index = material_indeces[obj['material']]
        collisions = obj['geometry'].intersections(ray)
        mask = collisions.dist < distance
        np.place(nearest_material_index, mask, material_index)
        np.copyto(distance, collisions.dist, where = mask)
        normal.copyfrom(collisions.norm, where = mask)
    
    # color pixels
    
    directional_lighting = scene['lighting']['directional'].unit()
    ambient_lighting = scene['lighting']['ambient']
    
    directional_component =  np.clip(directional_lighting.dot(normal) * -1, 0, 1)
    lighting = (directional_component * (1 - ambient_lighting) + ambient_lighting)
    
    matte_component = V3(0.0, 0.0, 0.0).repeat(area)
    reflective_component = V3(0.0, 0.0, 0.0).repeat(area)
    
    reflectivity = np.repeat([0.0], area)
    
    for i, material_name in enumerate(material_list):
        mask = (nearest_material_index == i)
        material = scene['materials'][material_name]
        matte_component.place(mask, material.color)
        np.place(reflectivity, mask, material.reflectivity)
    
    reflective_mask = (reflectivity > 0.0)
    if bounce > 0 and reflective_mask.any():
        position_set = (ray[0] + ray[1] * distance).extract(reflective_mask)
        incident_set = ray[1].extract(reflective_mask)
        normal_set = normal.extract(reflective_mask)
        reflected_set = incident_set - normal_set.unit() * incident_set.dot(normal_set) * 2
        reflective_camera = CameraPrecomputed((position_set, reflected_set))
        reflective_component_set = render(reflective_camera, scene, bounce - 1)
        reflective_component.place(reflective_mask, reflective_component_set)
    
    raster += matte_component * lighting * (1.0 - reflectivity)
    raster += reflective_component * reflectivity
    
    return raster