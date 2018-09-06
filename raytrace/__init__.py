import numpy as np

from .vector import *
from .camera import *
from .geometry import *
from .material import *

def render(camera, scene):
    area = camera.resolution[0] * camera.resolution[1]
    
    nearest_object = np.repeat([-1], area)
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
    
    # compute nearest intersections
    for i, obj in enumerate(scene['objects']):
        (obj_distance, obj_normal) =  obj['geometry'].intersections(camera.rays())
        mask = obj_distance < distance
        np.place(nearest_object, mask, i)
        np.copyto(distance, obj_distance, where = mask)
        normal.copyfrom(obj_normal, where = mask)
    
    # compute pixel colors
    for i, obj in enumerate(scene['objects']):
        mask = (nearest_object == i)
        nn = normal.extract(mask)
        directionalLighting =  np.clip(scene['lighting']['directional'].unit().dot(nn) * -1, 0, 1)
        ambience = scene['lighting']['ambient']
        lighting = (directionalLighting * (1 - ambience) + ambience)
        raster.place(mask, obj['material'].getApparentColor(lighting))
    
    return raster