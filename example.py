from raytrace import *
from PIL import Image


resolution = (1000, 1000)
origin = V3(0, -3, 0.5)
lookat = V3(4, 0, 0)
direction = (lookat - origin).unit()
dimensions = (3, 3)



camera = CameraOrthogonal(origin, direction, dimensions, resolution)

scene = {
    'objects': [
        {
            'geometry': Difference(
                Intersection(
                    Sphere(V3(4, 0, 0), 1),
                    Sphere(V3(4, -1, 0), 1)
                ),
                Sphere(V3(4, -1, 0), 0.5)
            ),
            'material': 'green'
        },
        {
            'geometry': Sphere(V3(4, -0.8, 0.8), 0.5),
            'material': 'checkered'
        },
        {
            'geometry': Ground(V3(0.0, 0.0, -20.0), V3(0.0, 0, 1.0)),
            'material': 'red'
        }
    ],
    'materials': {
        'red': UniformMaterial(V3(1, 0, 0)),
        'green': UniformMaterial(V3(0, 1, 0)),
        'blue': UniformMaterial(V3(0, 0, 1)),
        'checkered': CheckeredMaterial(
            UniformMaterial(V3(1, 1, 1)),
            UniformMaterial(V3(0, 0, 0)),
            scale = 0.5
        )
    },
    'lighting': {
        'ambient': 0.1,
        'directional': V3(1, 1, -1)
    }
}

raster = render(camera, scene) * 255

rgb = [
    Image.fromarray(raster.x.reshape(resolution).astype(np.uint8), "L"),
    Image.fromarray(raster.y.reshape(resolution).astype(np.uint8), "L"),
    Image.fromarray(raster.z.reshape(resolution).astype(np.uint8), "L")
]
Image.merge("RGB", rgb).save("example.png")