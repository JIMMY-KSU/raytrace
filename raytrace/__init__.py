import numpy as np

from .vector import *
from .camera import *
from .geometry import *
from .material import *

def render(camera, scene):
    area = camera.resolution[0] * camera.resolution[1]
    
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
        (obj_distance, obj_normal) =  obj['geometry'].intersections(camera.rays())
        mask = obj_distance < distance
        np.place(nearest_material_index, mask, material_index)
        np.copyto(distance, obj_distance, where = mask)
        normal.copyfrom(obj_normal, where = mask)
    
    # compute pixel colors
    for i, material in enumerate(material_list):
        mask = (nearest_material_index == i)
        nn = normal.extract(mask)
        directionalLighting =  np.clip(scene['lighting']['directional'].unit().dot(nn) * -1, 0, 1)
        ambience = scene['lighting']['ambient']
        lighting = (directionalLighting * (1 - ambience) + ambience)
        material = scene['materials'][material]
        raster.place(mask, material.getApparentColor(lighting))
    
    return raster