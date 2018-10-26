from .render import *
from .vector import *

class AmbientLight:
    
    def __init__(self, brightness = 1, color = V3(1, 1, 1)):
        self.brightness = brightness
        self.color = color
    
    def illuminate(self, scene, collisions):
        return (self.color * self.brightness).repeat(collisions.area)


class DirectionalLight:
    
    def __init__(self, direction, brightness = 1, color = V3(1, 1, 1)):
        self.direction = direction.unit()
        self.brightness = brightness
        self.color = color
    
    def illuminate(self, scene, collisions):
        parallel = np.clip(self.direction.dot(collisions.norm) * -1, 0, 1)
        unshadowed = self.color * self.brightness * parallel
        
        shadow_ray = Ray(
            collisions.incd,
            (self.direction * -1).repeat(collisions.area)
        )
        shadow_collisions = collide(collisions.area, shadow_ray, scene)
        shadow_mask = (shadow_collisions.incd.normsq() == np.inf)
        
        return unshadowed * shadow_mask


class PointLight:
    
    def __init__(self, position, brightness = 1, color = V3(1, 1, 1)):
        self.position = position
        self.brightness = brightness
        self.color = color
    
    def illuminate(self, scene, collisions):
        displacements = self.position - collisions.incd
        shadow_ray = Ray(
            collisions.incd,
            displacements.unit()
        )
        shadow_collisions = collide(collisions.area, shadow_ray, scene)
        distsq = displacements.normsq()
        shadow_mask = ((self.position - shadow_collisions.incd).normsq() > distsq)
        
        parallel = np.clip(shadow_ray.v.dot(collisions.norm), 0, 1)
        unshadowed = self.color * self.brightness * parallel / distsq
        
        return unshadowed * shadow_mask
