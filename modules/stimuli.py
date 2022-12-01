# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 15:52:43 2020

@author: smolina
"""

# Importing packages
from __future__ import division
from psychopy import visual,core,logging,event
from psychopy.hardware import keyboard
from psychopy.visual.windowwarp import Warper
from matplotlib import pyplot as plt # For some checks
import pyglet.gl as GL
import PyDAQmx as daq
import numpy as np
import copy
import time
import cv2

from modules.helper import *
from modules.exceptions import StopExperiment, MicroscopeException, StimulusTimeExceededException, GlobalTimeExceededException
from modules import config

def field_flash(bg_ls,fg_ls,stim_texture,noise_arr,stimdict, epoch, window, global_clock, duration_clock,
                outFile,out, stim_obj,dlpOK, viewpos, data,taskHandle = None,
                lastDataFrame = 0, lastDataFrameStartTime = 0):

    """field_flash:

    This stimulus function draws a stim_obj (circle or a rectangle) of a given size.
    Usually, the the stimulus covers the whole screen, eliciting a full-field flash

    bg_ls: defines the level of luminance of the screen per epoch
    fg_ls:  defines the level of luminance of the stim_obj per epoch
    pos: defines the posisitonb of the stim_obj. [0,0] is the center of the screen
    tau: duration in seconds for fg presentation
    duration: entire duration in seconds (bg + fg)
    framerate: is the refresh rate of the monitor

    """


    win = window
    win.colorSpace = 'rgb' # R G B values in range: [-1, 1]
    win.color= bg_ls[epoch] # Background for selected epoch

    # circle attributes
    if stimdict["stimtype"][epoch] == "C":
        stim_obj.radius= stimdict["radius"][epoch]

    # rectangle attributes
    elif stimdict["stimtype"][epoch] == "R":
        stim_obj.width = stimdict["width"][epoch]
        stim_obj.height = stimdict["height"][epoch]
        

    # set timing
    tau = stimdict["tau"][epoch]
    duration = stimdict["duration"][epoch]
    framerate = config.FRAMERATE
    frame_shift = 0 # For stim_obj texture
    start_frame = 0 # For stim_obj texture

    # "number"  and "interSpace" attributes are present in only some stimuli
    stim_obj_ls, space_ls = [], [] # Only implemented for vertical and horizontal bars (see bar.ori)
    try:
        stim_number = int(stimdict["number"][epoch])
        inter_space = stimdict["interSpace"][epoch]
        for i in range(stim_number):
            if stim_number == 1:
                space_ls.append(0.0)
            else:

                space_ls.append(inter_space * i)
                #stim_obj.pos[0] = stim_obj.pos[0]-space_ls[i] # For sister objects
            stim_obj_ls.append(stim_obj)
    except:
        stim_number = 1
        stim_obj_ls.append(stim_obj)
        space_ls.append(0.0)


    # generating texture for luminance values
    if stimdict["stimtype"][epoch] == 'NC':

        wave_lenght = len(stim_texture[0]) # Lenght of the original wave
        noise_arr = noise_arr[1,:,:] # Making lenghts of signal and noise the same

        long_wave = stim_texture[0] # Initialyzing the array with 1 signal wave
        for wave in stim_texture:
            long_wave = np.append(long_wave,wave)
        stim_texture = long_wave

        long_noise_arr = noise_arr[0] # Initialyzing the array with 1 noise wave
        for noise_wave in noise_arr:
            long_noise_arr = np.append(long_noise_arr,noise_wave)
        noise_arr = long_noise_arr

        circle_texture = stim_texture + noise_arr # Final texture (=lum values) to apply
        frequency =  stimdict["frequency"][epoch] # Frequency to change lum values
        framerate = config.FRAMERATE # Screen frame rate
        frame_shift = round((wave_lenght * frequency)/framerate)


    # Information to print
    BG=  ((bg_ls[epoch][2]+1)/2)/(63.0/255.0) # Scaling values back to a range of [0 1]
    FG= ((fg_ls[epoch][2]+1)/2)/(63.0/255.0)  # Scaling values back to a range of [0 1]
    print(f'BG level: {BG}')
    if BG == 0.0:
        pass
    else:
        # The WC calculation only makes sense if the win values is being showed
        if stimdict["tau"][epoch] == stimdict["duration"][epoch]:
            pass
        else:
            WC = (FG-BG)/BG
            print(f'FG level: {FG}')
            print(f'WC: {WC}')


    # As long as duration, draw the stimulus
    # Reset epoch timer
    duration_clock = global_clock.getTime()
    reset_bar_position = False

    for frameN in range(int(duration*framerate)):
        # fast break on key (ESC) pressed
        if len(event.getKeys(['escape'])):
            raise StopExperiment
        
        #Resetting sisters stim_obj possition for next frame
        if reset_bar_position: # event avoided for first iteration (frame)
            stim_obj.pos[0] = stim_obj.pos[0] + sum(space_ls)

        # As long as tau, draw FOREGROUND (> sign direction)
        if global_clock.getTime()-duration_clock >= tau:
            # For each bar object specified by the user (see "bar.number")
            for i,stim_obj in enumerate(stim_obj_ls):
                try:
                    stim_obj.pos = (stimdict['x_center'][epoch],stimdict['y_center'][epoch])
                except:
                    stim_obj.pos = (0,0)
                # stim_obj attributes for drawing fg
                if stimdict["stimtype"][epoch] == 'NC':
                    stim_obj.fillColor = [-1, -1,circle_texture[start_frame]] # in RGB
                    stim_obj.lineColor= [-1, -1,circle_texture[start_frame]] # in RGB
                    #stim_obj.radius= stimdict["radius"][epoch]
                else:
                    stim_obj.pos[0] = stim_obj.pos[0]-space_ls[i] # For sister objects
                    #stim_obj.pos[1] = stim_obj.pos[1]-space_ls[i] # For sister objects
                    stim_obj.fillColor = fg_ls[epoch]
                    stim_obj.lineColor= fg_ls[epoch]

                #print(stim_obj.pos[0])
                stim_obj.draw()



        elif global_clock.getTime()-duration_clock < tau:
            # stim_obj attributes for drawing BACKGROUND
            stim_obj.fillColor = bg_ls[epoch]
            stim_obj.lineColor= bg_ls[epoch]
            stim_obj.draw()


        # store Output
        out.tcurr = global_clock.getTime()
        out.yPos = time.time()
        out.xPos = float(stim_obj.pos[0])

        # NIDAQ check, timing check and writeout
        # quick and dirty fix to run stimulus on dlp without mic
        if not stimdict["MAXRUNTIME"] == 0:
            (out.data,lastDataFrame, lastDataFrameStartTime) = \
            check_timing_nidaq(dlpOK,stimdict["MAXRUNTIME"],global_clock,
                               taskHandle,data,lastDataFrame,
                               lastDataFrameStartTime)
        write_out(outFile,out)

        out.framenumber = out.framenumber +1
        win.flip() # swap buffers
        reset_bar_position = True
        start_frame = start_frame + frame_shift

        # #SavingMovieFrames
        # win.getMovieFrame() #Frames are stored in memory until a saveMovieFrames() command is issued.

    return (out, lastDataFrame, lastDataFrameStartTime)


def standing_stripes_random(bg_ls,fg_ls,stimdict, epoch, window, global_clock, duration_clock, outFile, out, bar, dlpOK, taskHandle=None, data=0, lastDataFrame=0, lastDataFrameStartTime=0):

    """standing_stripes_random:

    Total number of bars depends on the minimum and maximum possible
    position of the bars along the x-axis, and also on the bar distance.
    Distance between bars is measured from the center of the bars,
    therefore with suitable bar width and bar distance, bars can overlap.
    The total number of different positions is equal to the number of bars.
    Positions are equally spaced but does not appear in order, in order to
    prevent adaptation. Instead, they are shuffled based on a default seed.


    """

    win = window
    win.color= bg_ls[epoch] # Background for selected epoch


    win.colorSpace = 'rgb' # R G B values in range: [-1, 1]
    bar.fillColor = fg_ls[epoch]
    bar.width = stimdict["bar.width"][epoch]
    bar.height = stimdict["bar.height"][epoch]
    bar.ori = stimdict["bar.orientation"][epoch]

    framerate = config.FRAMERATE
    scr_width = win.scrWidthCM
    scr_distance = win.scrDistCM
    position_seed = config.SEED
    bar_duration = int(stimdict["bar.duration"][epoch] * framerate)
    bg_duration = int(stimdict["bg.duration"][epoch] * framerate)


    #Single bar, random locations
    print(f'Getting random positions')
    if stimdict["bar.orientation"][epoch] == 0:
        positions = position_x(stimdict, epoch, screen_width=scr_width, distance=scr_distance, seed=position_seed)
    elif stimdict["bar.orientation"][epoch] == 90:
        positions = position_y(stimdict, epoch, screen_width=scr_width, distance=scr_distance, seed=position_seed)

    bar_no = len(positions)
    epoch_duration = (bg_duration + bar_duration) * bar_no

    out.boutInd = 0 # Re-initialize the boutInd (single bar flash) for every epoch
    out.boutInd = out.boutInd + 1

    counter = [0, 0]
    for n in range(epoch_duration):

        if len(event.getKeys(['escape'])):
            raise StopExperiment

        if counter[0] < bar_duration:
            if stimdict["bar.orientation"][epoch] == 0:
                bar.pos = [positions[counter[1]], 0]
                out.xPos = float(positions[counter[1]])
                out.yPos = 0.0
                #bar.pos = [0, 0] #Seb temporary

            elif stimdict["bar.orientation"][epoch] == 90:
                bar.pos = [0,positions[counter[1]]]
                out.yPos = float(positions[counter[1]])
                out.xPos = 0.0

            #print(bar.pos)
            bar.draw()


            # out.xPos = float(positions[counter[1]])
            counter[0] += 1

        elif bar_duration <= counter[0] < (bg_duration + bar_duration - 1):
            out.xPos = float("NaN")
            out.yPos = float("NaN")
            counter[0] += 1

        elif counter[0] == (bg_duration + bar_duration - 1):
            out.xPos = float("NaN")
            out.yPos = float("NaN")
            counter[0] = 0
            counter[1] += 1 # For the next stripe
            out.boutInd = out.boutInd + 1

        out.tcurr = global_clock.getTime()

         # quick and dirty fix to run stimulus on dlp without mic
        if not stimdict["MAXRUNTIME"] == 0:
            (out.data, lastDataFrame, lastDataFrameStartTime) = check_timing_nidaq(dlpOK, stimdict["MAXRUNTIME"], global_clock, taskHandle, data, lastDataFrame, lastDataFrameStartTime)
        write_out(outFile, out)
        out.framenumber = out.framenumber + 1

        win.flip()
        # #SavingMovieFrames
        # win.getMovieFrame() #Frames are stored in memory until a saveMovieFrames() command is issued.

    return (out, lastDataFrame, lastDataFrameStartTime)

def drifting_stripe(exp_Info,bg_ls,fg_ls,stimdict, epoch, window, global_clock, duration_clock, outFile,out, bar,dlpOK, viewpos, data,taskHandle = None, lastDataFrame = 0, lastDataFrameStartTime = 0):
    """drifting_stripe:
    """
    #print(f' FUNCTION STARTS: {global_clock.getTime()}')
    win = window
    win.color= bg_ls[epoch]  # Background for selected epoch
    bar.fillColor = fg_ls[epoch]
    bar.width = stimdict["bar.width"][epoch]
    bar.height = stimdict["bar.height"][epoch]
    bar.ori = stimdict["bar.orientation"][epoch]
    # set timing
    tau = stimdict["tau"][epoch]
    duration = stimdict["duration"][epoch]
    direction = stimdict["direction"][epoch]
    framerate = config.FRAMERATE

    # Size of your actual window (in degrees of visual angle)
    scr_width = win.scrWidthCM
    scr_distance = win.scrDistCM

    # Setting edge positions
    direction = set_edge_position_and_direction(bar,scr_width,scr_distance,exp_Info,direction)
    print(f'Direction: {direction}')

    # "bar.initPos" attribute are present in only some stimuli
    try:
        init_pos  = stimdict["bar.initPos"][epoch]
        print(f'Initial position: {init_pos} ')
    except:
        init_pos = 0.0 # In case the stim input file does not have an initial position, put it to the center

    # "bar.number"  and "bar.interSpace" attributes are present in only some stimuli
    bar_ls, space_ls = [], [] # Only implemented for vertical and horizontal bars (see bar.ori)
    try:
        bar_number = int(stimdict["bar.number"][epoch])
        inter_space = stimdict["bar.interSpace"][epoch]
        for i in range(bar_number):
            bar_ls.append(bar)
            if bar_number == 1:
                space_ls.append(0.0)
            else:
                space_ls.append(inter_space * i)

    except:
        bar_number = 1
        bar_ls.append(bar)
        space_ls.append(0.0)


    # As long as duration, draw the stimulus
    # Reset epoch timer
    reset_bar_position = False
    duration_clock = global_clock.getTime()
    for frameN in range(int(duration*framerate)): # for seconds*100fps
        # fast break on key (ESC) pressed
        if len(event.getKeys(['escape'])):
            raise StopExperiment

        #Resetting sisters bar possition for next frame
        if reset_bar_position: # event avoided for first iteration (frame)
            if bar.ori == 0:
                if direction == "right":
                    bar.pos[0] = bar.pos[0] + sum(space_ls)
                elif direction == "left":
                    bar.pos[0] = bar.pos[0] - sum(space_ls)
            elif bar.ori == 90 :
                if direction == "up":
                    bar.pos[1] = bar.pos[1] + sum(space_ls)
                elif direction == "down":
                    bar.pos[1] = bar.pos[1] - sum(space_ls)

        # As long as tau, draw FOREGROUND (> sign direction)
        if global_clock.getTime()-duration_clock >= tau:

            # For each bar object specified by the user (see "bar.number")
            for i,bar in enumerate(bar_ls):
                if bar.ori == 0 :
                    bar.pos = (bar.pos[0], init_pos) # Fixing Y position to a desire initial position
                    # Move Stimulus with velocity per second
                    if direction == "right": # For bars moving to the right along the x-axis
                        bar.pos[0] = bar.pos[0]-space_ls[i] # For sister bars
                        bar.pos += ((stimdict["velocity"][epoch]/framerate)/bar_number,0.0)
                    elif direction == "left": # For bars moving to the left along the x-axis
                        bar.pos[0] = bar.pos[0]+space_ls[i] # For sister bars
                        bar.pos -= ((stimdict["velocity"][epoch]/framerate)/bar_number,0.0)
                elif bar.ori == 90 :
                    bar.pos = (init_pos,bar.pos[1]) # Fixing X position to a desire initial position
                    # Move Stimulus with velocity per second
                    if direction == "up": # For bars moving to the right along the x-axis
                        bar.pos[1] = bar.pos[1]-space_ls[i] # For sister bars
                        bar.pos += (0.0,(stimdict["velocity"][epoch]/framerate)/bar_number)
                    elif direction == "down": # For bars moving to the left along the x-axis
                        bar.pos[1] = bar.pos[1]+space_ls[i] # For sister bars
                        bar.pos -= (0.0,(stimdict["velocity"][epoch]/framerate)/bar_number)
                elif bar.ori ==  45:
                    # Move Stimulus with velocity per second
                    bar.width = stimdict["bar.width"][epoch] * np.sqrt(2) # Correcting size for diagonals
                    if direction == "left-up": # For bars moving to the right along the x-axis
                        bar.pos -= ((stimdict["velocity"][epoch]/framerate)/bar_number*np.sqrt(2),0.0)
                    elif direction == "right-down": # For bars moving to the left along the x-axis
                        bar.pos += ((stimdict["velocity"][epoch]/framerate)/bar_number*np.sqrt(2),0.0)
                elif bar.ori == 135:
                    # Move Stimulus with velocity per second
                    bar.width = stimdict["bar.width"][epoch] * np.sqrt(2) # Correcting size for diagonals
                    if direction == "right-up": # For bars moving to the right along the x-axis
                        bar.pos += (0.0,(stimdict["velocity"][epoch]/framerate)/bar_number*np.sqrt(2))
                    elif direction == "left-down": # For bars moving to the left along the x-axis
                        bar.pos -= (0.0,(stimdict["velocity"][epoch]/framerate)/bar_number*np.sqrt(2))
                #print(bar.pos)
                bar.draw()
        # store Output
        out.tcurr = global_clock.getTime()
        out.xPos = float(bar.pos[0])
        out.yPos = time.time()
        out.theta = float(stimdict["velocity"][epoch])
        # NIDAQ check, timing check and writeout
        # quick and dirty fix to run stimulus on dlp without mic
        if not stimdict["MAXRUNTIME"] == 0:
            (out.data,lastDataFrame, lastDataFrameStartTime) = check_timing_nidaq(dlpOK,stimdict["MAXRUNTIME"],global_clock,taskHandle,data,lastDataFrame,lastDataFrameStartTime)
        write_out(outFile,out)
        out.framenumber = out.framenumber +1
        win.flip() # swap buffers
        reset_bar_position = True
        # #SavingMovieFrames
        # win.getMovieFrame() #Frames are stored in memory until a saveMovieFrames() command is issued.
    #print(f'FUNCTION ENDS: {global_clock.getTime()}')
    return (out, lastDataFrame, lastDataFrameStartTime)



def stim_noise(bg_ls,stim_texture,stimdict, epoch, window, global_clock, duration_clock, outFile, out, noise, dlpOK, taskHandle=None, data=0, lastDataFrame=0, lastDataFrameStartTime=0):

    """stim_noise:


    """
    win = window
    win.color= bg_ls[epoch]  # Background for selected epoch
    win.colorSpace = 'rgb'

    # set timing
    framerate = config.FRAMERATE


    # Size of your actual window (in the units chosen, normally degrees)
    scr_width = win.scrWidthCM
    scr_distance = win.scrDistCM
    maxhorang = max_angle_from_center(scr_width, scr_distance) # From the middle to one side
    maxhorang  = maxhorang  * 2 # Full screen

    # Grating (here noise) attributes
    noise.size= (maxhorang, maxhorang)
    noise.sf = 1/maxhorang

    tex_duration = int(stimdict['texture.duration'][epoch] * framerate) # Duration in frame number
    tex_count = int(stimdict['texture.count'][epoch])
    hor_size = np.sqrt(stimdict['texture.hor_size'][epoch])
    vert_size = np.sqrt(stimdict['texture.vert_size'][epoch])
    hor_size  = int(hor_size)
    vert_size  = int(vert_size)


    texture = copy.deepcopy(stim_texture) # An independent copy
    texture *= 63.0/255.0 # convert from 8 bit depth to 6 bit depth.
    texture = texture * 2 - 1 # the *2-1 part converts the color space [0,1] -> [-1,1]


    for count,t in enumerate(texture):
        for frameN in range(tex_duration):
            if len(event.getKeys(['escape'])):
                raise StopExperiment

            #Geeting RGB values for the texture
            rgb_t = numpy.zeros((t.shape[0],t.shape[1],3), dtype=np.float32)
            rgb_t[:,:,0] = -1 # All R value to -1
            rgb_t[:,:,1] = -1 # All G value to -1
            rgb_t[:,:,2] = t
            # for i in range(1):
            #     rgb_t[:,:,i+1] = t # Setting G and B values

            # noise.tex = t
            noise.tex = rgb_t
            noise.draw()

            out.tcurr = global_clock.getTime()
            out.theta = count
            if not stimdict["MAXRUNTIME"] == 0:
                (out.data, lastDataFrame, lastDataFrameStartTime) = check_timing_nidaq(dlpOK, stimdict["MAXRUNTIME"], global_clock,taskHandle,data,lastDataFrame,lastDataFrameStartTime)
            write_out(outFile, out)

            out.framenumber = out.framenumber + 1

            win.flip()

    return (out, lastDataFrame, lastDataFrameStartTime)



def noisy_grating(_useNoise,_useTex,viewpos,bg_ls,stim_texture,noise_arr,stimdict, epoch, window, global_clock, duration_clock, outFile, out, grating, dlpOK, taskHandle=None, data=0, lastDataFrame=0, lastDataFrameStartTime=0):

    """noisy_grating:

    Noise is generated by summing random numbers from an Uniform distribution
    to a sinusoidal wave. The mean noise distribution is = 0 and the std
    is a variable value that depends on the level of noise to achieve.
    The noise std is chosen based on the desired signal to noise ratio (SNR)
    with respect to a sinusoidal signal as follows:
        noise_std = mean(sine_signal)/SNR


    """
    win = window
    win.color= bg_ls[epoch]  # Background for selected epoch
    win.colorSpace = 'rgb'
    # win.flip() # draw background
    # win.flip() # present background


    # set timing
    framerate = config.FRAMERATE
    tau = stimdict["tau"][epoch]
    duration = int(stimdict['duration'][epoch] * framerate) # Duration in frame number


    # Size of your actual window (in the units chosen, normally degrees)
    scr_width = win.scrWidthCM
    scr_distance = win.scrDistCM
    maxhorang = max_angle_from_center(scr_width, scr_distance) # From the middle to one side
    maxhorang  = maxhorang  * 2 # Full screen


    # grating attributes
    if _useTex:
        grating.tex = stim_texture

    grating.sf = 1/stimdict['sWavelength'][epoch]
    grating.size= (maxhorang, maxhorang)
    try:
        grating.ori = stimdict["orientation"][epoch]
        direction = int(stimdict["direction"][epoch]) # Direction of the moving grating: either +1 or -1
        print('Orientation: {}, Direction: {}'.format( grating.ori,direction))
    except:
        print('Stim without specified direction and orientation. Default: 0 deg and left')
        grating.ori = 0
        direction = 1

    _phaseValue = (stimdict['velocity'][epoch]/(framerate*stimdict['sWavelength'][epoch])) * direction

    # mask
    try:
        if stimdict['mask'][epoch]:
            grating.mask = 'circle'
            grating.pos = [stimdict['pos.x'][epoch],stimdict['pos.y'][epoch]]
            grating.size = stimdict['mask.size'][epoch]
    except:
        pass


    # variable to store
    if stimdict["stimtype"][epoch] == 'noisygrating':
        output_value = stimdict['SNR'][epoch]
        print('{} SNR'.format(output_value))
    elif stimdict["stimtype"][epoch] == 'lumgrating':
        output_value = stimdict['lum'][epoch]
        print('{} lum'.format(output_value))
    elif stimdict["stimtype"][epoch] == 'TFgrating':
        output_value = float(stimdict['velocity'][epoch])/stimdict['sWavelength'][epoch] # Temporal frequency
        print('{} hz'.format(output_value))


    # Reset epoch timer
    duration_clock = global_clock.getTime()
    max_tex_value = (2*(63.0/255.0))-1 # Max value in stim_texture after scaling
    min_tex_value = -1 # Min value in stim_texture after scaling
    for frameN in range(duration):
            if len(event.getKeys(['escape'])):
                raise StopExperiment
            # noise.draw()   #The noise object is currently NOT IN USE
            if _useNoise:
                # Adding noise to the original signal
                grating_texture = stim_texture + noise_arr[frameN]
                # Adjusting noisy values out of the range

                grating_texture[np.where(grating_texture> max_tex_value)] = max_tex_value
                grating_texture[np.where(grating_texture<min_tex_value)] = min_tex_value
                grating.tex = grating_texture

            # After tau, change the phase of grating (motion)
            if global_clock.getTime()-duration_clock >= tau:
                grating.phase += _phaseValue
            grating.draw()


            out.tcurr = global_clock.getTime()
            out.theta = output_value
            if not stimdict["MAXRUNTIME"] == 0:
                (out.data, lastDataFrame, lastDataFrameStartTime) = check_timing_nidaq(dlpOK, stimdict["MAXRUNTIME"], global_clock,taskHandle,data,lastDataFrame,lastDataFrameStartTime)
            write_out(outFile, out)

            out.framenumber = out.framenumber + 1

            win.flip()

            ##SavingMovieFrames
            #win.getMovieFrame() #Frames are stored in memory until a saveMovieFrames() command is issued.

    # Checking what we actually present per frame
    fig1,ax = plt.subplots(2,2)
    ax[0, 0].plot(grating.tex.T)
    ax[0, 0].axhline(y = max_tex_value, color = 'k', linestyle = ':', label = "max")
    ax[0, 0].axhline(y = min_tex_value, color = 'k', linestyle = '--', label = "min")
    ax[0, 0].set_title('{}%MC_{}_SNR'.format(stimdict['michealson.contrast'][-1]*100,output_value))
    ax[0, 1].plot(grating.tex[0])
    ax[0, 1].set_title('{}%MC_{}_SNR'.format(stimdict['michealson.contrast'][-1]*100,output_value))
    ax[1, 0].plot(grating.tex[0:3,:].T)
    ax[1, 1].imshow(grating.tex,cmap=plt.get_cmap('gray'))

    # Computing contrast of the image based on proportion of pixels with min and max values
    total_num_pixels = grating.tex.size
    max_count = np.count_nonzero(grating.tex== np.max(grating.tex))
    max_ratio = round((max_count/ total_num_pixels),3)
    min_count = np.count_nonzero(grating.tex== np.min(grating.tex))
    min_ratio = round((min_count/ total_num_pixels),3)

    contrast = (max_ratio + min_ratio)/2 # mean
    ax[1, 1].set_title('Contrast: %f' % (round(contrast,3)*100))



    output_dir = 'F:\\SebastianFilesExternalDrive\\Science\\PhDAGSilies\\2pData Python_data\\Trash'
   #  fig1.savefig('{}\\{}%MC_{}_SNR.png'.format(output_dir,stimdict['michealson.contrast'][-1]*100,output_value))
   # fig1.savefig('{}\\{}%MC_{}_SNR.pdf'.format(output_dir,stimdict['michealson.contrast'][-1]*100,output_value))

    return (out, lastDataFrame, lastDataFrameStartTime)




def dotty_grating(_useNoise,_useTex,viewpos,bg_ls,stim_texture,stimdict, epoch, window, global_clock, duration_clock, outFile, out, grating,dots, dlpOK, taskHandle=None, data=0, lastDataFrame=0, lastDataFrameStartTime=0):

    """dotty_grating:

    Noise is generated by drawing dots of the same lumiance as the mean
    luminance of either an squarewave or sinusoidal grating. Different levels
    of noise are achieved by changing the number of dots on the screen.
    Dots have a coherence of 100% a e behaviour takenfrom Scase et al’s (1996)
    categorie: ‘position’, meaning that dots take a random position every frame.
    Dots have a size of 5 pisels and a life time of 3 frames.


    """


    win = window
    win.color= bg_ls[epoch]  # Background for selected epoch
    win.colorSpace = 'rgb'


    # set timing
    framerate = config.FRAMERATE
    tau = stimdict["tau"][epoch]
    duration = int(stimdict['duration'][epoch] * framerate) # Duration in frame number


    # Size of your actual window (in the units chosen, normally degrees)
    scr_width = win.scrWidthCM
    scr_distance = win.scrDistCM
    maxhorang = max_angle_from_center(scr_width, scr_distance) # From the middle to one side
    maxhorang  = maxhorang  * 2 # Full screen


    # grating attributes
    if _useTex:
        grating.tex = stim_texture
    grating.sf = 1/stimdict['sWavelength'][epoch]
    grating.size= (maxhorang, maxhorang)
    _phaseValue = stimdict['velocity'][epoch]/(framerate*stimdict['sWavelength'][epoch])



    # dots attributes
    dots.refreshDots()
    dots.nDots= int(stimdict['nDots'][epoch])
    dots.dotSize= int(stimdict['dotSize'][epoch])
    dots.speed=int(stimdict['dotSpeed'][epoch])
    dots.speed= 0
    dots.fieldSize= (maxhorang, maxhorang)

    # Reset epoch timer
    duration_clock = global_clock.getTime()
    for frameN in range(duration):
            if len(event.getKeys(['escape'])):
                raise StopExperiment

            # dots.draw()
            dots.setAutoDraw(True)
            # After tau, change the phase of grating (motion)
            if global_clock.getTime()-duration_clock >= tau:
                grating.phase += _phaseValue
                # grating.setPhase(stimdict['setPhase'][epoch],'+') #Deprecated
            grating.draw()


            out.tcurr = global_clock.getTime()
            out.theta = dots.nDots
            if not stimdict["MAXRUNTIME"] == 0:
                (out.data, lastDataFrame, lastDataFrameStartTime) = check_timing_nidaq(dlpOK, stimdict["MAXRUNTIME"], global_clock,taskHandle,data,lastDataFrame,lastDataFrameStartTime)
            write_out(outFile, out)

            out.framenumber = out.framenumber + 1

            win.flip()

    dots.setAutoDraw(False)
    return (out, lastDataFrame, lastDataFrameStartTime)

print("Module 'stimuli' imported")
