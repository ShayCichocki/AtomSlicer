"""Small test meshes for the Metal port (mm, plate at z=0, centred in x,y)."""
import os, trimesh
HERE = os.path.dirname(os.path.abspath(__file__))

def place(m):
    m.apply_translation([-m.bounds[:, 0].mean(), -m.bounds[:, 1].mean(), -m.bounds[0, 2]])
    assert m.is_watertight and m.is_volume
    return m

cube = place(trimesh.creation.box(extents=[20, 20, 20]))
cube.export(os.path.join(HERE, 'cube20.stl'))
blk = trimesh.creation.box(extents=[30, 20, 12])
hole = trimesh.creation.cylinder(radius=3.0, height=20, sections=64)
holed = place(blk.difference(hole))
holed.export(os.path.join(HERE, 'holed_block.stl'))
print('cube', cube.volume, 'holed', holed.volume)
