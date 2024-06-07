#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from collections import defaultdict
from random import seed
import PyDAQmx as daq
import numpy as np
import datetime
import os
import imageio

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
            self._viewpos = np.genfromtxt(filename, dtype=None)
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
        expInfo = '%s %s %s\n' % (exp_Info["Experiment"],exp_Info["User"],exp_Info["TSeries_ID"] )
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
        index = np.zeros((no_epochs,1))
        index = index.astype(int)
        for ii in range(0,no_epochs):
            index[ii] = ii

    elif randomize == 1.0:
        # shuffle epochs randomly, except epoch 0
        # every 2nd epochchoose == 0
        index = np.zeros((no_epochs-1,1))
        index = index.astype(int)

        for ii in range(0,no_epochs-1):
            index[ii] = ii+1

        # np.random.seed(config.SEED)
        np.random.shuffle(index) # Actual shuffling

    elif randomize == 2.0:
        # shuffle epochs randomly
        index = np.zeros((no_epochs,1))
        index = index.astype(int)

        for ii in range(no_epochs):
            index[ii] = ii

        # np.random.seed(config.SEED)
        np.random.shuffle(index) # Actual shuffling


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

    max_ang = np.arctan((screen_width/2) / distance)
    max_ang = abs(np.degrees(max_ang))


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

    max_ang = np.arctan((screen_width) / distance)
    max_ang = (abs(np.degrees(max_ang)))


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


    xpos = np.arange(xmin, xmax, bar_distance)
    seeder = np.random.RandomState(seed)
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

    ypos = np.arange(ymin, ymax, bar_distance)
    ypos = ypos*-1 #-1 to move downswards with the stimulus
    seeder = np.random.RandomState(seed)
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

def create_movie_from_png(input_folder, output_path, fps=24):
    # Get all .png files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.png')]
    image_files.sort()  # Sort files to ensure the correct order

    # Create a list to store images
    images = []

    # Read each image file and append it to the images list
    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        image = imageio.imread(image_path)
        images.append(image)

    # Write the images to a movie file
    with imageio.get_writer(output_path, fps=fps) as writer:
        for image in images:
            writer.append_data(image)

def edge_postitioning_and_width(bar,scr_width,scr_distance,angle):
    
    " edge postion for an arbitrary angular direction from the perspective of an observer 0 deg is rigth, 180 left, 90 up 270 down"
    
    # define cuadrant of edge initial location

    x=round(1*np.cos(np.deg2rad(angle)),4) # the minus is necesary for allowing the correct direction of movement (for example 0 deg is right movement then needs to start at the left)
    if x==0:
        x_pos=0
    elif x>0:
        x_pos=1
    else:
        x_pos=-1

    y=round(1*np.sin(np.deg2rad(angle)),4)
    if y==0:
        y_pos=0
    elif y>0:
        y_pos=1
    else:
        y_pos=-1
    location_vector = np.array([x_pos,y_pos])
    maximum_angle=max_angle_from_center(scr_width, scr_distance)
    #maximum_diag_angle=np.sqrt(2*(maximum_angle**2)) # this is the distance in angles from the center of the screen to the corner
    init_pos=np.array([maximum_angle,maximum_angle])*location_vector
    print(f'maximum_angle: {maximum_angle}')
    #init_pos=np.array([30,30])*location_vector
    # find the bar width for the orientation 

    if np.abs(x)>np.abs(y):
        hypotenuse= maximum_angle/np.abs(x)

    elif np.abs(x)<np.abs(y):
        hypotenuse= maximum_angle/np.abs(y)

    else:
        hypotenuse= np.sqrt((maximum_angle**2)*2)
    
    bar.width=(2*(hypotenuse))+10
    print(f'bar width: bar.width')

    # move the bar so the edge lands either in an edge or the corner of the screen
    shift_x= (bar.width/2)*x
    shift_y= (bar.width/2)*y
    span=bar.width
    #init_pos = np.array([init_pos[0]+shift_x,init_pos[1]+shift_y]) 
    #init_pos = np.array([0 , 0])
    return init_pos#,span

def find_step_decomposition(angle_dir,step):
    """ find the vector decomposition of an unit vector that describes the direction of movement of an edge from its angular direction
        for example. for angle_dir =45 the movement vector will be (-cos(45),-sin(45))"""
    angle_rad = np.deg2rad(angle_dir)
    return np.array([-round(np.cos(angle_rad),4),-round(np.sin(angle_rad),4)])*step # the minus value is due to the fact that the fly has a fipped view of the scene

def reflect_angle(angle):
    
    """ find the reflected angle across the x axis. this is needed when a mirror is in the stimulus projection path, since the mirror flips 
    the image across the x axis"""
    # check if angle is in [0, 360)
    if angle < 0 or angle >= 360:
        raise ValueError("The angle must be in the range [0, 360)")

        
    if angle <= 180:
        return 180 - angle

    # if angle is in [180, 360), the reflected angle is 540 - angle
    else:
        return 540 - angle
    
def random_persistent_behavior_vector(seeds,frames,choices_dur):
    
    """ this function uses a random seed to create a list of random numbers from a choice list 
        the choice list represents the range of durations possible from where values are drawn based on an
        uniform distribution
        
        output is a list of numbers which sum is equal or higher than the number of frames"""

    if frames>20000:
        raise Exception ('seems like this is too many frames. consider if this is necessary ')

    final_vectors = []
    
    for local_seed in seeds:
        local_vector = []
        np.random.seed(local_seed)
        while np.sum(local_vector)<frames:
            local_vector.append(np.random.choice(choices_dur))

        final_vectors.append(local_vector)
    return final_vectors

def random_persistent_values(persistent_behavior_vectors,seeds,frames,possible_values,size):

    """ using the output of random_persistent_behavior_vector(seeds,frames,choices_dur) this function 
    draws possible values from an uniform distribution to populate or modify a noise stimulus
    
    seeds: random seeds to draw values (as many as persistent behavior vectors are required
    frames: lenght of stimulus in frames
    possible values: the set of values from which is possible to choose
    size: the size of a frame of output, if youre building a video, then size is (x,y) dimensions
          if you are for example modifying a frames by shifting one dimension, then size is (1)... 
          
    output: chosen random values based on persistent behavior vector and a uniform distribution"""

    output_values=[]
    
    for vector,local_seed in zip(persistent_behavior_vectors,seeds):
        if len(size)==2:
            local_outputvals = np.zeros((np.sum(vector),size[0],size[1]))
        else:
            local_outputvals = np.zeros((np.sum(vector)))
        np.random.seed(local_seed)
        count=0
        for ix,repeats in enumerate(vector):
            if len(size)==2:
                value_movement = np.random.choice(possible_values,size=(size[0],size[1]))
                local_outputvals[count:count+repeats,:,:] = value_movement[np.newaxis,:,:]
            else:          
                value_movement = np.random.choice(possible_values,size=(1))
                local_outputvals[count:count+repeats] = value_movement
            count=count+repeats

        if len(size)==2:
            output_values.append(local_outputvals[:frames,:,:])
        else:
            output_values.append(local_outputvals[:frames])
    return output_values


def helper_cumsum__wrap(array_towrap,max,min):
    """
    Calculates a wrapped cumulative sum of a 2xN array.
    Parameters:
    a (numpy.ndarray): A 2xN matrix where each column represents a movement in x and y directions.
    Returns:
    numpy.ndarray: A 2xN matrix representing the cumulative sum with wrapped boundaries.
    """
    if array_towrap.shape[0] != 2:
        raise ValueError("Input array must be 2xN.")

    cum_sum = np.zeros_like(array_towrap)
    for i in range(array_towrap.shape[1]):
        cum_sum[:, i] = array_towrap[:, i] if i == 0 else cum_sum[:, i-1] + array_towrap[:, i]

        # Apply wrapping for each dimension
        for dim in range(2):
            while cum_sum[dim, i] > max:
                cum_sum[dim, i] = -max + (cum_sum[dim, i] - max)
            while cum_sum[dim, i] < -min:
                cum_sum[dim, i] = max + (cum_sum[dim, i] + max)

    return cum_sum


def guassian_distributed_lums(seed, lum_vector_size):  #Pradeep
    """ The funtion uses a trucated normal distribution to generate lumimance values used by the
    gaussian_flash function.
    Parameters for the truncated normal distribution are as follows:
        a_trunc = lower limit to cut-off the distribution tails
        b_trunc =  upper limit to cut-off the distribution tails
        loc = mean value to center the distribution around
        scale = the std dev or the spread of the gaussian distributed stimuli
    Seeds: to set the seed for random num generator
    lum_vector_size: set the length of gaussian distributed stimuli

    output: a vector of size determined by lum_vector_size"""

    np.random.seed(seed)
    a_trunc = 0
    b_trunc = 1
    loc = 0.6
    scale = 0.35
    a, b = (a_trunc - loc) / scale, (b_trunc - loc) / scale

    lum_gaussian = truncnorm.rvs(a, b, loc = loc , scale = scale, size = lum_vector_size)

    return(lum_gaussian)
