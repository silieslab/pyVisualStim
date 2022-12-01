from numpy import array, radians, zeros, hypot, tan
from psychopy import monitors


def _degPerspective2pix(vertices, pos, win):
    return deg2pixPerspective(array(pos) + array(vertices), win.monitor,
                   correctFlat=True)

def deg2pixPerspective(degrees, monitor, correctFlat=False):
    """Convert size in degrees to size in pixels for a given Monitor object
    """
    # get monitor params and raise error if necess
    scrWidthCm = monitor.getWidth()
    scrSizePix = monitor.getSizePix()
    if scrSizePix is None:
        msg = "Monitor %s has no known size in pixels (SEE MONITOR CENTER)"
        raise ValueError(msg % monitor.name)
    if scrWidthCm is None:
        msg = "Monitor %s has no known width in cm (SEE MONITOR CENTER)"
        raise ValueError(msg % monitor.name)

    cmSize = deg2cmPerspective(degrees, monitor, correctFlat)
    #print cmSize * scrSizePix[0] / float(scrWidthCm)

    return cmSize * scrSizePix[0] / float(scrWidthCm)


def deg2cmPerspective(degrees, monitor, correctFlat=False):
    """Convert size in degrees to size in pixels for a given Monitor object.
    If `correctFlat == False` then the screen will be treated as if all
    points are equal distance from the eye. This means that each "degree"
    will be the same size irrespective of its position.
    If `correctFlat == True` then the `degrees` argument must be an Nx2 matrix
    for X and Y values (the two cannot be calculated separately in this case).
    With `correctFlat == True` the positions may look strange because more
    eccentric vertices will be spaced further apart.
    """
    # check we have a monitor
    if not hasattr(monitor, 'getDistance'):
        msg = ("deg2cm requires a monitors.Monitor object as the second "
               "argument but received %s")
        raise ValueError(msg % str(type(monitor)))
    # get monitor dimensions
    dist = monitor.getDistance()
    # check they all exist
    if dist is None:
        msg = "Monitor %s has no known distance (SEE MONITOR CENTER)"
        raise ValueError(msg % monitor.name)
    if correctFlat:
        rads = radians(degrees)
        cmXY = zeros(rads.shape, 'd')  # must be a double (not float)
        if rads.shape == (2,):
            x, y = rads
            cmXY[0] = hypot(dist, tan(y) * dist) * tan(x)
            cmXY[1] = hypot(dist, tan(x) * dist) * tan(y)
            # edit this: change cmXY[1:0]
        elif  rads.shape[0] >= 4 and rads.shape[1] == 2:
            # It doesn't paint it like that :( 
            ##########################
            monitor_height = monitors.Monitor('testMonitor').getSizePix()[1] # bad, hard-coded, in cm
            monitor_width = monitors.Monitor('testMonitor').getSizePix()[0]
            monitor_width_cm = monitors.Monitor('testMonitor').getWidth()
            pix_size = monitor_width_cm / monitor_width
            monitor_height_cm = monitor_height * pix_size
            
            # X
            #cmXY[0, 0] = hypot(hypot(dist, monitor_height_cm), tan(rads[0, 1]) * dist) * tan(rads[0, 0])-10
            cmXY[0, 0] = hypot(dist, monitor_height_cm) * tan(rads[0, 0])
            #cmXY[1, 0] = hypot(dist, tan(rads[1, 1]) * dist) * tan(rads[1, 0])
            cmXY[1, 0] = dist * tan(rads[1, 0])
            #cmXY[2, 0] = hypot(dist, tan(rads[2, 1]) * dist) * tan(rads[2, 0])
            cmXY[2, 0] = dist * tan(rads[2, 0])
            #cmXY[3, 0] = hypot(hypot(dist, monitor_height_cm), tan(rads[3, 1]) * dist) * tan(rads[3, 0])
            cmXY[3, 0] = hypot(dist, monitor_height_cm) * tan(rads[3, 0])
            
            # Y
            cmXY[0, 1] = hypot(hypot(dist, monitor_height_cm), tan(rads[0, 0]) * dist) * tan(rads[0, 1])
            cmXY[1, 1] = hypot(dist, tan(rads[1, 0]) * dist) * tan(rads[1, 1])
            cmXY[2, 1] = hypot(dist, tan(rads[2, 0]) * dist) * tan(rads[2, 1])
            cmXY[3, 1] = hypot(hypot(dist, monitor_height_cm), tan(rads[3, 0]) * dist) * tan(rads[3, 1])
            
            print (cmXY)
        #elif len(rads.shape) > 1 and rads.shape[1] == 2:
            #cmXY[:, 0] = hypot(dist, tan(rads[:, 1]) * dist) * tan(rads[:, 0])
            #cmXY[:, 1] = hypot(dist, tan(rads[:, 0]) * dist) * tan(rads[:, 1])
            #print cmXY
            #pass
        else:
            msg = ("If using deg2cm with correctedFlat==True then degrees "
                   "arg must have shape [N,2], not %s")
            raise ValueError(msg % (repr(rads.shape)))
        # derivation:
        #    if hypotY is line from eyeball to [x,0] given by
        #       hypot(dist, tan(degX))
        #    then cmY is distance from [x,0] to [x,y] given by
        #       hypotY * tan(degY)
        #    similar for hypotX to get cmX
        # alternative:
        #    we could do this by converting to polar coords, converting
        #    deg2cm and then going back to cartesian,
        #    but this would be slower(?)
        return cmXY
    else:
        # the size of 1 deg at screen centre
        return array(degrees) * dist * 0.017455
