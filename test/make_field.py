"""Synthetic FXFD1 external field for testing -F: layer normal tilts about y with height."""
import struct, sys, numpy as np
out = sys.argv[1]
origin = np.array([-5.0, -5.0, -5.0], np.float32); sp = 2.0; dims = (40, 20, 40)
with open(out, 'wb') as f:
    f.write(b'FXFD1'); f.write(origin.tobytes()); f.write(struct.pack('<f', sp)); f.write(struct.pack('<3I', *dims))
    cells = np.zeros((dims[2], dims[1], dims[0], 8), np.float32)   # x fastest
    z = origin[2] + sp * (np.arange(dims[2]) + 0.5)
    th = 0.5 * np.clip(z / 40.0, 0, 1)                              # up to ~29 deg
    cells[..., 0] = np.sin(th)[:, None, None]; cells[..., 2] = np.cos(th)[:, None, None]
    cells[..., 3] = 1.0; cells[..., 6] = 1.0; cells[..., 7] = 0.0
    f.write(cells.tobytes())
