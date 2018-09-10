from .vector import *

class CameraOrthogonal:
    
    def __init__(self, position, direction, dimensions, resolution):
        self.position = position
        self.direction = direction
        self.dimensions = dimensions
        self.resolution = resolution
    
    def area(self):
        return self.resolution[0] * self.resolution[1]
        
    def rays(self):
        width = self.resolution[0]
        height = self.resolution[1]
        area = width * height
        
        # unit directions on screen
        right = self.direction.cross(V3(0, 0, 1)).unit()
        down =  self.direction.cross(right).unit()
        
        # unit directions in pixel units
        du = right.scale(self.dimensions[0] / self.resolution[0])
        dv = down.scale(self.dimensions[1] / self.resolution[1])
        
        upper_left = (self.position
                   + right.scale(-0.5 * self.dimensions[0])
                   + down.scale(-0.5 * self.dimensions[1]))
        
        # displacements relative to upper left corner per pixel
        us = du.scale(
            np.tile(
                np.arange(0.5, width  + 0.5, 1),
                height
            )
        )
        vs = dv.scale(
            np.repeat(
                np.arange(0.5, height + 0.5, 1),
                width
            )
        )
        
        positions = upper_left + us + vs
        
        # all rays are parallel for an orthogonal camera
        directions = self.direction.repeat(area)
        
        return (positions, directions)

class CameraPrecomputed:
    
    def __init__(self, precomputed_rays):
        self.precomputed_rays = precomputed_rays
    
    def area(self):
        return len(self.precomputed_rays[0].x)
    
    def rays(self):
        return self.precomputed_rays