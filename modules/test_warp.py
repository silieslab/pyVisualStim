import numpy as np 

mon_width_cm = 29
mon_height_cm = 15
dist_cm = 7
eyepoint  = [0.5,1]
isCylindrical = False

nverts = (128 - 1) * (128 - 1) * 4

# eye position in cm
xEye = eyepoint[0] * mon_width_cm
yEye = eyepoint[1] * mon_height_cm

# create vertex grid array, and texture coords
# times 4 for quads
vertices = np.zeros(
    ((128 - 1) * (128 - 1) * 4, 2), dtype='float32')
tcoords = np.zeros(
    ((128 - 1) * (128 - 1) * 4, 2), dtype='float32')

equalDistanceX = np.linspace(0, mon_width_cm, 128)
equalDistanceY = np.linspace(0, mon_height_cm, 128)

# vertex coordinates
x_c = np.linspace(-1.0, 1.0, 128)
y_c = np.linspace(-1.0, 1.0, 128)
x_coords, y_coords = np.meshgrid(x_c, y_c)

x = np.zeros(((128), (128)), dtype='float32')
y = np.zeros(((128), (128)), dtype='float32')

x[:, :] = equalDistanceX - xEye
y[:, :] = equalDistanceY - yEye
y = np.transpose(y)

r = np.sqrt(np.square(x) + np.square(y) + np.square(dist_cm))

azimuth = np.arctan(x / dist_cm)
altitude = np.arcsin(y / r)

# calculate the texture coordinates
if isCylindrical:
    tx = dist_cm * np.sin(azimuth)
    ty = dist_cm * np.sin(altitude)
else:
    tx = dist_cm * (1 + x / r) - dist_cm
    ty = dist_cm * (1 + y / r) - dist_cm

# prevent div0
azimuth[azimuth == 0] = np.finfo(np.float32).eps
altitude[altitude == 0] = np.finfo(np.float32).eps

# the texture coordinates (which are now lying on the sphere)
# need to be remapped back onto the plane of the display.
# This effectively stretches the coordinates away from the eyepoint.

if isCylindrical:
    tx = tx * azimuth / np.sin(azimuth)
    ty = ty * altitude / np.sin(altitude)
else:
    centralAngle = np.arccos(
        np.cos(altitude) * np.cos(np.abs(azimuth)))
    # distance from eyepoint to texture vertex
    arcLength = centralAngle * dist_cm
    # remap the texture coordinate
    theta = np.arctan2(ty, tx)
    tx = arcLength * np.cos(theta)
    ty = arcLength * np.sin(theta)

u_coords = tx / mon_width_cm + 0.5
v_coords = ty / mon_height_cm + 0.5

# loop to create quads
vdex = 0

for y in xrange(0, 128 - 1):
    for x in xrange(0, 128 - 1):
        index = y * (128) + x

        vertices[vdex + 0, 0] = x_coords[y, x]
        vertices[vdex + 0, 1] = y_coords[y, x]
        vertices[vdex + 1, 0] = x_coords[y, x + 1]
        vertices[vdex + 1, 1] = y_coords[y, x + 1]
        vertices[vdex + 2, 0] = x_coords[y + 1, x + 1]
        vertices[vdex + 2, 1] = y_coords[y + 1, x + 1]
        vertices[vdex + 3, 0] = x_coords[y + 1, x]
        vertices[vdex + 3, 1] = y_coords[y + 1, x]

        tcoords[vdex + 0, 0] = u_coords[y, x]
        tcoords[vdex + 0, 1] = v_coords[y, x]
        tcoords[vdex + 1, 0] = u_coords[y, x + 1]
        tcoords[vdex + 1, 1] = v_coords[y, x + 1]
        tcoords[vdex + 2, 0] = u_coords[y + 1, x + 1]
        tcoords[vdex + 2, 1] = v_coords[y + 1, x + 1]
        tcoords[vdex + 3, 0] = u_coords[y + 1, x]
        tcoords[vdex + 3, 1] = v_coords[y + 1, x]

        vdex += 4
        

#elf.createVertexAndTextureBuffers(vertices, tcoords)