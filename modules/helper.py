#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from collections import defaultdict
import PyDAQmx as daq
import numpy
import datetime

from modules.exceptions import MicroscopeException, StimulusTimeExceededException, GlobalTimeExceededException
from modules import config




class Viewpositions(object):
    """ Reads the Viewpositions file and stores its values

        :param filename: The viewpositions.txt file
        :type filename: path

    """

    __slots__ = ['x','y','width','height','_viewpos']

    def __init__(self,filename):
        self._viewpos = []

        try:
            self._viewpos = numpy.genfromtxt(filename, dtype=None)
        except ValueError:
            print ('Viewpositions could not be read. Error msg:')
            raise

        self.x = self._viewpos[0]
        self.y = self._viewpos[1]
        self.width = self._viewpos[2]
        self.height = self._viewpos[3]



class Output(object):

    """ The Output class contains all data which will be written to the output file

    """
    __slots__ = ['framenumber','tcurr','boutInd','epochchoose','xPos','yPos','rand_intensity','theta','data']
    def __init__(self):
        self.framenumber = 0
        self.tcurr = 0
        self.boutInd = 0
        self.epochchoose = 0
        self.xPos = 0
        self.yPos = 0
        self.theta = 0
        self.data = 0

    def save_outfile(self,location):
        """
        OLD, deprecated
        :param location: Location where files should be stored
        :type location: path
        """
        time = datetime.datetime.now()
        outfile_name = "%s\\%s_%d%d_%d.txt" %(location,config.OUTFILE_NAME,time.hour,time.minute,time.second)

        outfile_temp = "%s\\%s.txt" %(location,config.OUTFILE_NAME)
        with open(outfile_temp) as source:
            with open(outfile_name, 'w') as dest:
                for line in source:
                    dest.write(line)


    def create_outfile_temp(self,location,path_stimfile,exp_Info):

        """
        :param location: Location where files should be stored
        :type location: path
        :param path_stimfile: Location and name of chosen stimfile
        :type path_stimfile: path

        """
        time = datetime.datetime.now()
        outfile_temp_name = "%s\\%s_%d%d_%d.txt" %(location,config.OUTFILE_NAME,time.hour,time.minute,time.second)
        outFile_temp = open(outfile_temp_name, 'w')
        expInfo = '%s %s %s\n' % (exp_Info["ExpName"],exp_Info["User"],exp_Info["Subject_ID"] )
        stimfile = '%s\n' % (path_stimfile)
        outFile_temp.write(expInfo)
        outFile_temp.write(stimfile)
        outFile_temp.write('frame,tcurr,boutInd,epoch,xpos,ypos,theta,data\n')

        return outFile_temp


class Stimulus(object):
    """ Takes a stimulus filename as input and creates a dictionary of lists with all stimulus attributes
        provided by the stimfile. If a list of epochs has only one element, the value will be extracted from the list and
        thus is directly usable.

        :param dict: dictionary containing the Stimulus attributes names as keys and as values lists containing
                    one element per epoch
        :type dict: defaultdict, default_factory = None

        """

    def __init__(self,filename):

        self.dict = self._read(filename)

    def _read(self,filename):

        """
        :param filename: Stimfile to read from
        :type filename: path

        """
        dict = defaultdict()

        with open(filename) as file:
            for line in file:
                curr_list = line.split()

                if not curr_list:
                    continue

                key = curr_list.pop(0)

                if len(curr_list) == 1 and not "Stimulus." in key:
                    try:
                        dict[key] = int(curr_list[0])
                    except ValueError:
                        dict[key] = curr_list[0]
                    continue

                if key.startswith("Stimulus."):
                    key = key[9:]

                    if key.startswith("stimtype"):
                        dict[key] = list(map(str, curr_list))
                    else:
                        dict[key] = list(map(float, curr_list))

        return dict


def write_main_setup(location,dlp_ok,MAXRUNTIME,exp_Info):

    """ Writes the meta_data file which logs global settings """

    # A temporary mainfile, containing data of last run
    time = datetime.datetime.now()
    mainfile_name_temp = f"{location}\\{config.METAFILE_NAME}_{time.hour}{time.minute}_{time.second}.txt"
    mainfile_temp = open(mainfile_name_temp, 'w')

    mainfile_temp.write("KEY,VALUE\n")
    mainfile_temp.write("useDLP,%d\n" % dlp_ok)
    mainfile_temp.write("MAXRUNTIME,%f\n" % round(MAXRUNTIME))
    for key,value in exp_Info.items():
        mainfile_temp.write(f"{key},{value}\n")

def save_main_setup(location):

    """ OLD, deprecated. Copies the current meta_data file to a timestamped meta_data file.
    This is execution specific."""

    # A permanently saved mainfile copy with time stemp
    time = datetime.datetime.now()
    mainfile_name = "%s\\_meta_data_%d_%d_%d_%d_%d_%d.txt" %(location,time.year,time.month,time.day,time.hour,time.minute,time.second)

    mainfile_temp = f"{location}\\{config.METAFILE_NAME}_{time.hour}{time.minute}_{time.second}.txt"
    with open(mainfile_temp) as source:
        with open(mainfile_name, 'w') as dest:
            for line in source:
                dest.write(line)


def write_out(outFile,out):

    """ Checks NIDAQ data and writes all out-data to output file.

    :param global_clock: global clock, will be written to output
    :type global_clock: core.Clock
    :param outFile: The output file
    :type outFile: A csv .txt file
    :param out: The output data
    :type out: `helper.Output`
    :param data: counter for microsope frames
    :type data: uInt32()
    :param lastDataFrame: last frame recorded by microsope
    :type lastDataFrame: uInt32()
    :param lastDataFrameStartTime: time the microsope started to record the last frame
    :type lastDataFrameStartTime: time

    :returns: The updated NIDAQ data and read in time.

    .. note::
        The NIDAQ arguments (data, lastDataFrame, lastDataFrameStartTime) are only needed if DLP is used

    .. note::
        The output file contains (framenumber,time, 0,epochchoose,xPos,0,theta = rotation,0)

    """
    # write out as a comma seperated file or tab separated
    outFile.write('%8d, %8.3f, %8d, %8d, %8.4f, %8.4f,%8.4f, %8d\n' %(out.framenumber,out.tcurr,out.boutInd,out.epochchoose,out.xPos,out.yPos,out.theta,out.data))
    # outFile.write('%8d %8.3f %8d %8d %8.4f %8.4f %8.4f %8d\n' %(out.framenumber,out.tcurr,out.boutInd,out.epochchoose,out.xPos,out.yPos,out.theta,out.data))

def check_timing_nidaq(dlpOK,stimdictMAXRUNTIME,global_clock,taskHandle = None,data = 0 ,lastDataFrame = 0 ,lastDataFrameStartTime = 0):
    """
    Reads in the microscope's signal, updates framenumber (in data) and checks if microscope still works in time.
    Furthermore it checks if a time constant has been exceeded (MAXRUNTIME's)

    :param dlpOK: Is DLP used (so, not in testmode)?
    :type dlpOK: boolean
    :param stimdictMAXRUNTIME: The stimulus' MAXRUNTIME
    :type stimdictMAXRUNTIME: int
    :param global_clock: global clock, will be written to output
    :type global_clock: core.Clock
    :param data: counter for microsope frames
    :type data: uInt32()
    :param lastDataFrame: last frame recorded by microsope
    :type lastDataFrame: uInt32()
    :param lastDataFrameStartTime: time the microsope started to record the last frame
    :type lastDataFrameStartTime: time

    :returns: The updated NIDAQ data and read in time.

    .. note::
        The NIDAQ arguments (data, lastDataFrame, lastDataFrameStartTime) are only needed if DLP is used
    """
    # check for DAQ Data
    if taskHandle != None:
        daq.DAQmxReadCounterScalarU32(taskHandle,1.0,daq.byref(data), None)

        if (lastDataFrame != data.value):
            lastDataFrame = data.value
            lastDataFrameStartTime = global_clock.getTime()

    # Irregular stop conditions:
    if dlpOK and (global_clock.getTime() - lastDataFrameStartTime > 1):
       raise MicroscopeException(lastDataFrame,lastDataFrameStartTime,global_clock.getTime())

    elif dlpOK and (global_clock.getTime() >= stimdictMAXRUNTIME):
        raise StimulusTimeExceededException(stimdictMAXRUNTIME,global_clock.getTime())

    elif global_clock.getTime() >= config.MAXRUNTIME:
        raise GlobalTimeExceededException(config.MAXRUNTIME,global_clock.getTime())

    return (data.value,lastDataFrame, lastDataFrameStartTime)

def shuffle_epochs(randomize,no_epochs):
    """Shuffles the epoch sequence according to the randomize option.

    :param randomize: 0 (don't shuffle), 1 (shuffle randomly, except 1st epoch), 2 (shuffle randomly).
    :type randomize: Integer
    :param no_epochs: Number of epochs in stimfile.
    :type no_epochs: Integer
    :returns: numpy integer array of shuffled epoch indices.

    """
    if randomize == 0.0:
        # dont shuffle epochs
        index = numpy.zeros((no_epochs,1))
        index = index.astype(int)
        for ii in range(0,no_epochs):
            index[ii] = ii

    elif randomize == 1.0:
        # shuffle epochs randomly, except epoch 0
        # every 2nd epochchoose == 0
        index = numpy.zeros((no_epochs-1,1))
        index = index.astype(int)

        for ii in range(0,no_epochs-1):
            index[ii] = ii+1

        # numpy.random.seed(config.SEED)
        numpy.random.shuffle(index) # Actual shuffling

    elif randomize == 2.0:
        # shuffle epochs randomly
        index = numpy.zeros((no_epochs,1))
        index = index.astype(int)

        for ii in range(no_epochs):
            index[ii] = ii

        # numpy.random.seed(config.SEED)
        numpy.random.shuffle(index) # Actual shuffling


    return index

def choose_epoch(index,randomize,no_epochs,current_index):
    """Shuffles the epoch sequence according to the randomize option.

    :param index: Array of shuffled epoch indices.
    :type index: Numpy int array
    :param randomize: 0 (choose next), 1 (every 2nd epoch choose epoch 0), 2 (choose next epoch).
    :type randomize: Integer
    :param no_epochs: Number of epochs in stimfile.
    :type no_epochs: Integer
    :param current_index: Current chosen index of index-array.
    :type current_indexs: Integer
    :returns: Integer of chosen epoch index


    """

    if randomize == 0.0 or randomize == 2.0:
        epochchoose = index.item((current_index,0))
        current_index = (current_index+1) % no_epochs
        print('---------------------')
        print('Presented epoch: {}'.format(epochchoose))

    elif randomize == 1.0:
         # every 2nd epochchoose == 0
        if (current_index % 2) == 1:
            epochchoose = index.item(int((current_index-1)/2),0)
            print('---------------------')
            print('Presented epoch: {}'.format(epochchoose))
        else:
            epochchoose = 0
            print('---------------------')
            print('Presented epoch: {}'.format(epochchoose))
        current_index = (current_index+1) % (2*(no_epochs-1))

    return (epochchoose, current_index)


def set_bgcol(lum,con):

    """ Sets background color according to luminance and contrast in stimfile

    It's calculated this way like::

        bgcol = lum*(1-con)

    """

    bgcol = lum*(1-con)

    background = [0]*3
    # the *2-1 part converts the color space [0,1] -> [-1,1]
    background[0] = (get_dlpcol(bgcol,'R')* config.COLOR_ON[0])*2-1
    background[1] = (get_dlpcol(bgcol,'G')* config.COLOR_ON[1])*2-1
    background[2] = (get_dlpcol(bgcol,'B')* config.COLOR_ON[2])*2-1


    return background

def set_fgcol(lum,con):

    """ Sets background color according to luminance and contrast in stimfile

    :param lum: the luminance defined py this epoch.
    :type lum: double
    :param con: the contrast defined py this epoch.
    :type con: double
    :returns: double


    It's calculated this way like::

        fgcol = lum*(1+con)

    """

    fgcol = lum*(1+con)

    foreground = [0]*3
    # the *2-1 part converts the color space [0,1] -> [-1,1]
    foreground[0] = (get_dlpcol(fgcol,'R')* config.COLOR_ON[0])*2-1
    foreground[1] = (get_dlpcol(fgcol,'G')* config.COLOR_ON[1])*2-1
    foreground[2] = (get_dlpcol(fgcol,'B')* config.COLOR_ON[2])*2-1


    return foreground

def set_intensity(epoch,value):

    """ Returns Intensity color
    It calls the function for gamma correction and 6-bit depth transformation

    :returns: list of 3 floats for RGB color space in range [-1,1]

    """

    intensity = [0] * 3
    # the *2-1 part converts the color space [0,1] -> [-1,1]
    intensity[0] = (get_dlpcol(value,'R')* config.COLOR_ON[0])*2-1
    intensity[1] = (get_dlpcol(value,'G')* config.COLOR_ON[1])*2-1
    intensity[2] = (get_dlpcol(value,'B')* config.COLOR_ON[2])*2-1

    return intensity

def get_dlpcol(DLPintensity,channel):

    """ Gamma correction

    This function uses some measured screen properties to correct light intensity.
    It also converts the values from 8 bit depth (0 to 255) to 6 bit depth (0 to 63)

    :param DLPintensity: the fore- or background color value.
    :type DLPintensity: double
    :param channel: the color channel green 'G' or blue 'B'
    :type channel: char
    :returns: double in [0,1].

    """

    # Some fixed - measured variables
    gamma_r = config.GAMMA_LS[0]
    scale_r = 1
    gamma_g = config.GAMMA_LS[1]
    scale_g = 1
    gamma_b = config.GAMMA_LS[2]
    scale_b = 1

    temp = 0
    # Applying the inverse of the current gamma
    if channel == 'R':
        temp = pow(DLPintensity/scale_r, 1/gamma_r)
    elif channel == 'G':
        temp = pow(DLPintensity/scale_g, 1/gamma_g)
    elif channel == 'B':
        temp = pow(DLPintensity/scale_b, 1/gamma_b)

    # keep the output in the closed interval [0, 1]
    if temp > 1:
        temp = 1
    if temp < 0:
        temp = 0


    # temp = DLPintensity; # debug line to use if we want to turn off gamma correction
    if config.MODE == 'patternMode':
        temp *= 63.0/255.0 # convert from 8 bit depth to 6 bit depth.
    else:
        temp *= 255.0/255.0 # keep the 8 bit depth


    return temp


def max_angle_from_center(screen_width, distance):

    """ Returns the angular extent of the screen from the center to the edge
    in degrees with respect to the fly

    Assumes that the subject lies on an axis which passes through the screen
    and that it is centered relative to the screen.
    Thus, a perpendicular line from the subject to the screen has a degree
    of zero. Returned value is always positive, therefore the other edge of
    the screen has the opposite sign of the returned value.
    It's calculated this way::

        angle = arctan((screen width/2) /distance of subject to the screen)

        which is the same as:

        angle = arctan(screen width /(2*distance of subject to the screen))


    :param screen_width: width of the screen
    :type screen_width: float
    :param distance: perpendicular distance of the subject to the screen
    :type distance: float
    :returns: float

    """

    max_ang = numpy.arctan((screen_width/2) / distance)
    max_ang = abs(numpy.degrees(max_ang))


    return max_ang

def max_angle_from_edge(screen_width, distance):

    """ Returns the angular extent of the screen from edge to edge in degrees
    with respect to the fly

    Assumes that the subject lies at the edge (e.g. top) of the screen.
    Thus, a perpendicular line from the subject to the screen edge has a degree
    of zero. Returned value is always positive.
    It's calculated this way::

         angle = (arctan((screen width) /distance of subject to the screen))

    :param screen_width: width of the screen
    :type screen_width: float
    :param distance: perpendicular distance of the subject to the screen
    :type distance: float
    :returns: float

    """

    max_ang = numpy.arctan((screen_width) / distance)
    max_ang = (abs(numpy.degrees(max_ang)))


    return max_ang


def position_x(stimdict, epoch, screen_width, distance, seed):

    """ Returns random position values on the x-axis.

    It makes use of a default seed value to make experiment replicable.
    If the maximum position in the stimulus file is higher than the angular
    extent, it makes the value in the stimulus file equal to the angular
    extent, therefore the stimulus is not placed outside of the screen.
    The same applies for the minimum porsition in the stimulus file.

    :param screen_width: width of the screen
    :type screen_width: float
    :param distance: perpendicular distance of the subject to the screen
    :type distance: float
    :param seed: seed to be used in the pseudo-random number generation
    :type seed: int
    :returns: NumPy integer array

    """
    xmax = stimdict["bar.xmax"][epoch]
    xmin = stimdict["bar.xmin"][epoch]
    bar_distance = stimdict["bar.distance"][epoch]

    hor_angle =  max_angle_from_center(screen_width, distance)
    # hor_extent = abs(hor_angle - (stimdict["bar.width"][epoch]))
    hor_extent = abs(hor_angle)

    if xmin < -hor_extent:
        xmin = -hor_extent

    if xmax > hor_extent:
        xmax = hor_extent


    xpos = numpy.arange(xmin, xmax, bar_distance)
    seeder = numpy.random.RandomState(seed)
    seeder.shuffle(xpos)

    return xpos

def position_y(stimdict, epoch, screen_width, distance, seed):

    """ Returns random position values on the y-axis.

    It makes use of a default seed value to make experiment replicable.
    If the maximum position in the stimulus file is higher than the angular
    extent, it makes the value in the stimulus file equal to the angular
    extent, therefore the stimulus is not placed outside of the screen.
    The same applies for the minimum porsition in the stimulus file.

    :param screen_width: width of the screen
    :type screen_width: float
    :param distance: perpendicular distance of the subject to the screen
    :type distance: float
    :param seed: seed to be used in the pseudo-random number generation
    :type seed: int
    :returns: NumPy integer array

    """


    ymax = stimdict["bar.ymax"][epoch]
    ymin = (stimdict["bar.ymin"][epoch])
    bar_distance = stimdict["bar.distance"][epoch]

    ver_angle = max_angle_from_center(screen_width, distance)
    # ver_extent = abs(ver_angle - (stimdict["bar.width"][epoch]))
    ver_extent = abs(ver_angle)

    if stimdict["pers.corr"][epoch] == 1:

        if ymin < -ver_extent:
            ymin = -ver_extent

        if ymax > ver_extent:
            ymax = ver_extent


    else:
        ver_angle = max_angle_from_center(screen_width, distance) # Vertical extent will max how we calculate the horizontal extent
        ver_extent = abs(ver_angle)
        ymin = -ver_extent
        ymax= ver_extent

    ypos = numpy.arange(ymin, ymax, bar_distance)
    ypos = ypos*-1 #-1 to move downswards with the stimulus
    seeder = numpy.random.RandomState(seed)
    seeder.shuffle(ypos)


    return ypos

def window_3masks(win,_monitor = 'testMonitor'):

    from psychopy import visual

    '''
    Draws mask of for a screen, so that the smaller screen is insede the big
    one like the scheme: the center (C) at the top of the small screen is in
    the center of the big one and that the big screen fills all the resolution
    in the x-axis.


                        #######################
                        #                     #
                        #                     #
                        #                     #
                        #     #####C#####     #
                        #     #         #     #
                        #     #         #     #
                        #     #         #     #
                        #######################


    Screen size of my laptop in psychopy = 1536,864
    '''

    ########################## Windows creation ############################
    win =  win
    win_mask_sizes =[[win.size[0],win.size[1]/2],[win.size[0]/4,win.size[1]],[win.size[0]/4,win.size[1]]]
    win_mask_positions =[[win.pos[0],win.pos[1]],[win.pos[0],win.pos[1]],[win.pos[0]+(3*(win.size[0]/4)),win.pos[1]]]

    win_mask_0 = visual.Window(fullscr = False, monitor='testMonitor',
                                size = win_mask_sizes[0], viewScale = [1,1],
                                screen = 0, pos = win_mask_positions[0],
                                color=[-1,-1,-1],useFBO = True,allowGUI=False,
                                viewOri = 0.0)

    win_mask_1 = visual.Window(fullscr = False, monitor='testMonitor',
                                size = win_mask_sizes[1], viewScale = [1,1],
                                screen = 0, pos = win_mask_positions[1],
                                color=[-1,-1,-1],useFBO = True,allowGUI=False,
                                viewOri = 0.0)

    win_mask_2 = visual.Window(fullscr = False, monitor='testMonitor',
                                size = win_mask_sizes[2], viewScale = [1,1],
                                screen = 0, pos = win_mask_positions[2],
                                color=[-1,-1,-1],useFBO = True,allowGUI=False,
                                viewOri = 0.0)

    win_mask_ls = [win_mask_0,win_mask_1,win_mask_2]

    return win_mask_ls

def set_edge_position_and_direction(bar,scr_width,scr_distance,exp_Info,direction):

    #Getting screen visual angles
    maxhorang = max_angle_from_center(scr_width, scr_distance)
    maxverang = max_angle_from_center(scr_width, scr_distance)
    #bar.width = maxhorang*2
    maxhorang = maxhorang * direction # x_position should be either 1 or -1
    maxverang = maxverang * direction # x_position should be either 1 or -1

    #Adjusting for the rectangle edge to be printed to screen edge
    #It considers the edge of the screen = (maxhorang)
    #and the half width of the rectangle = (stimdict["spacing"][epoch]/2)
    if maxhorang  > 0:
        shift_pos = (maxhorang) + (bar.width/2)
    elif maxhorang  < 0:
        shift_pos = (maxhorang) - (bar.width/2)

    # Adusting the initial position of the stimulus based on bar orientation
    if bar.ori == 0: #vertical bar
        bar.pos = (shift_pos, 0.0)
    elif bar.ori == 90: #horizontal bar
        bar.pos = (0.0,shift_pos)
    elif bar.ori == 45:
        bar.pos = (-shift_pos, shift_pos)
    elif bar.ori == 135:
        bar.pos = (shift_pos, shift_pos)
    else:
        print('Bar orientation not compatible')

    if exp_Info['WinMasks']:
        scr_width = scr_width/2
        #Recalculating only the maxhorang
        maxhorang = max_angle_from_center(scr_width, scr_distance)
        maxhorang = maxhorang * direction # x_position should be either 1 or -1
        #bar.width = maxhorang*2

        # Recheck these calculations
        if direction  > 0:
            shift_pos = (maxhorang) + (bar.width/2)
            shift_x_pos = maxverang + (bar.width/2)
            shift_y_pos = maxverang + (bar.width/2)
        elif direction  < 0:
            shift_pos = (maxhorang) - (bar.width/2)
            shift_x_pos = maxverang - (bar.width/2)
            shift_y_pos = maxverang - (bar.width/2)

        # Adusting the initial position of the stimulus based on bar orientation
        if bar.ori == 0: #vertical bar
            bar.pos = (shift_pos, 0.0)
            print(f'Going lateral from: {bar.pos}')
        elif bar.ori == 90: #horizontal bar
            if direction  > 0:
                bar.pos = (0.0,(bar.width/2)) #bar will go down from here
                print(f'Going down from: {bar.pos}')
            elif direction  < 0:
                bar.pos = (0.0,shift_y_pos) #bar will go up from here
                print(f'Going up from: {bar.pos}')

        # IMPORTANT 45 and 135 agles still have a bug. Rethink!
        elif bar.ori == 45:
            if direction  > 0:
                bar.pos = (-shift_x_pos, (bar.width/2))
                print(f'Going right-down from: {bar.pos}')
            elif direction  < 0:
                bar.pos = (-shift_x_pos, shift_y_pos)
                print(f'Going left-up from: {bar.pos}')
        elif bar.ori == 135:
            if direction  > 0:
                print(f'Going left-down from: {bar.pos}')
                bar.pos = (shift_x_pos, (bar.width/2))
            elif direction  < 0:
                bar.pos = (shift_pos, shift_y_pos)
                print(f'Going rigth-up from: {bar.pos}')
        else:
            print('Bar orientation not compatible')


    #Assigning directionality to the epoch
    if bar.pos[0] > 0 and bar.pos[1] < 0:
         direction = "left-up"
    elif bar.pos[0] < 0 and bar.pos[1] > 0:
         direction = "right-down"
    elif bar.pos[0] < 0 and bar.pos[1] < 0:
         direction = "right-up"
    elif bar.pos[0] > 0 and bar.pos[1] > 0:
         direction = "left-down"
    elif bar.pos[0] < 0:
        direction = "right"
    elif bar.pos[0] > 0:
        direction = "left"
    elif bar.pos[1] < 0:
        direction = "up"
    elif bar.pos[1] > 0:
        direction = "down"

    return direction
