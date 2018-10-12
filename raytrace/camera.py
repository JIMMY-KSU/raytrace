from .vector import *


class CameraPerspective:
    
    def __init__(self, position, direction, dimensions, resolution):
        self.position = position
        self.direction = direction.unit()
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
                   + down.scale(-0.5 * self.dimensions[1])
                   + self.direction)
        
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
        
        positions = self.position.repeat(area)
        directions = (upper_left + us + vs - self.position).unit()
        
        return Ray(positions, directions)


class CameraOrthogonal:
    
    def __init__(self, position, direction, dimensions, resolution):
        self.position = position
        self.direction = direction.unit()
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
        
        return Ray(positions, directions)


class CameraPanoramic:
    
    def __init__(self, pos, long_min, long_max, lat_min, lat_max, resolution):
        self.pos = pos
        self.long_min = long_min * np.pi / 180.0
        self.long_max = long_max * np.pi / 180.0
        if self.long_min > self.long_max:
            self.long_max += 2 * np.pi
        self.lat_min = lat_min * np.pi / 180.0
        self.lat_max = lat_max * np.pi / 180.0
        self.resolution = resolution
    
    def area(self):
        return self.resolution[0] * self.resolution[1]
    
    def rays(self):
        width = self.resolution[0]
        height = self.resolution[1]
        area = width * height
        
        dlong = (self.long_max - self.long_min) / width
        dlat = (self.lat_max - self.lat_min) / height
        longs = np.tile(
            np.linspace(self.long_max - 0.5 * dlong, self.long_min + 0.5 * dlong, width),
            height
        )
        lats = np.repeat(
            np.linspace(self.lat_max - 0.5 * dlat, self.lat_min + 0.5 * dlat, height),
            width
        )
        
        return Ray(
            self.pos.repeat(area),
            V3(
                np.cos(lats) * np.cos(longs),
                np.cos(lats) * np.sin(longs),
                np.sin(lats)
            )
        )


class CameraPrecomputed:
    
    def __init__(self, precomputed_rays):
        self.precomputed_rays = precomputed_rays
    
    def area(self):
        return len(self.precomputed_rays)
    
    def rays(self):
        return self.precomputed_rays