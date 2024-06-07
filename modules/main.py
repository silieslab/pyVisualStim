

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import psychopy
from psychopy import visual,core,logging,event, gui, monitors
from psychopy.visual.windowwarp import Warper # perspective correction
from matplotlib import pyplot as plt # For some checks
import PyDAQmx as daq
# The PyDAQmx module is a full interface to the NIDAQmx ANSI C driver.
# It imports all the functions from the driver and imports all the predefined
# constants.
# This provides an almost one-to-one match between C and Python code
import pyglet.window.key as key
import numpy as np
import h5py
import datetime
import time
import sys
import copy
import os

from modules.helper import *
from modules.exceptions import *
from modules import config
from  modules import stimuli

#%%
def main(path_stimfile):
    """
        This function handles the window,logging, nidaq and played stimulus ...

        It also contains the monitor information you might want to edit.
        Press **Esc** to end the presentation immediately.
        Press **any key** to end the presentation after this epoch.

        :param path_stimfile: the path to the stimulus txt file
        :type path_stimfile: str

        .. note::
        The stimulus attributes specified in the txt file will change several
        options, such as: prespective correction, randomization of epoch order,
        stimulus type, etc. For more info, check each stimulus txt file and the
        function that controls it specified under Stimulus.stimtype


    """

##############################################################################
################################### GUI INPUTS ###############################
##############################################################################
    # Question: Put it on DLP ?
    dlp = gui.Dlg(title=u'Light Crafter', pos=None, size=None, style=None,
                  labelButtonOK=u' Yes ', labelButtonCancel=u' No ', screen=-1)
    dlp.addText('Want to use the DLP?')
    dlp_ok = dlp.show()

    #Messages
    mssg = gui.Dlg(title="Messages")
    mssg.addText('In the next box, you will define the basic experimental parameter')
    mssg.addText('\nVIEWPOINTS have a range from 1 to -1. Eg., x = 0.5, y =0.5 refers to the screen center')
    mssg.addText("WARP options for prespective correction: ‘spherical’, ‘cylindrical, ‘warpfile’ or None")
    mssg.addText('\nPress OK to continue')
    mssg.show()
    if mssg.OK == False:
        core.quit()  # user pressed cancel

    # Store info about the experiment session
    exp_Info = {'Experiment': config.ID_DICT['EXP_NAME'],'User': config.ID_DICT['USER_ID'], 'Subject_ID': config.ID_DICT['SUBJECT_ID'],
                'TSeries_ID': f"{config.ID_DICT['SUBJECT_ID']}-{config.ID_DICT['TSERIES_NUMBER']}",'Genotype': config.ID_DICT['GENOTYPE'],
                'Condition' : config.ID_DICT['CONDITION'],'Stimulus' : config.ID_DICT['STIMULUS_ID'], 'Age' : config.ID_DICT['AGE'],
                'Sex' : config.ID_DICT['SEX'],'ViewPoint_x': config.VIEWPOINT_X, 'ViewPoint_y':config.VIEWPOINT_Y, 'Warp': config.WARP,
                'Projector_mode': config.MODE, 'saving_movie_frames': config.SAVE_MOVIE}

    dlg = gui.DlgFromDict(dictionary=exp_Info, sortKeys=False, title="Experimental parameters")

    if dlg.OK == False:
        core.quit()  # user pressed cancel

    _time = datetime.datetime.now()
    exp_Info['date'] = "%d%d%d_%d%d_%d" %(_time.year,_time.month,
                                                _time.day,_time.hour,
                                                _time.minute,_time.second)

    exp_Info['psychopyVersion'] = psychopy.__version__
    exp_Info['Frame_rate'] = config.FRAMERATE
    exp_Info['Screen_distance'], exp_Info['Screen_width'] = config.DISTANCE, config.SCREEN_WIDTH
    exp_Info['WinMasks'] = config.WIN_MASK

 ##############################################################################
 #######################Settings for DLP Pattern Mode##########################
 ##############################################################################
    if exp_Info['Projector_mode'] == 'patternMode':
        _viewScale = [1,1/2]
    else:
        _viewScale = [1,1]
 ##############################################################################

    # Create output file
    out = Output()
    outFile = out.create_outfile_temp(config.OUT_DIR,path_stimfile,exp_Info)

    # Read coonfig settings
    MAXRUNTIME, framerate = config.MAXRUNTIME, config.FRAMERATE
    fname = path_stimfile
    current_index = 0
    epoch = 0 #First epoch of the stimulus file
    stop = False

    # Read stimulus file
    stimulus = Stimulus(fname)
    stimdict = stimulus.dict
    #stimdict["PERSPECTIVE_CORRECTION"] = 1 #Temporary until changing all stimuli

    #Adjusting old stim names to new ones
    for s, stimtype in enumerate(stimdict["stimtype"]):
    # Functions that draw the different stimuli
        if stimtype == "stripe(s)":
            stimdict["stimtype"][s] = "SSR"

        elif stimtype == "circle":
            stimdict["stimtype"][s] = "C"

        elif stimtype == "noisy_circle":
            stimdict["stimtype"][s] = "NC"

        elif stimtype == "driftingstripe":
            stimdict["stimtype"][s] = "DS"
        
        elif stimdict["stimtype"][epoch] == "arbitrarydriftingstripe": 
            stimdict["stimtype"][s] = "ADS"
            
        elif stimtype == "noise":
            stimdict["stimtype"][s] = "N"

        elif stimtype == "grating":
            stimdict["stimtype"][s] = "G"
        
        elif stimtype == "lumgrating":
            stimdict["stimtype"][s] = "G"

        elif stimtype == "dottygrating":
            stimdict["stimtype"][s] = "DG"

    # Read Viewpositions
    viewpos = Viewpositions(config.VIEWPOS_FILE)
    _width, _height = viewpos.width[0], viewpos.height[0]
    _xpos, _ypos = viewpos.x[0], viewpos.y[0]

##############################################################################
############### Creating the window where to draw your screen ################
##############################################################################

    # IMPORTANT: in order to use warper for perspective correction,
    # useFBO = True is important for our window
    # What the useFBO does is render your window to an 'offscreen' window first
    # before drawing that to the 'back buffer'and that process allows us to do
    # some fancy transforms on the whole window when we then flip()

    #Initializing the window as a dark screen (color=[-1,-1,-1])
    if dlp.OK: #Using the projector
        # Initializing screen
        mon = monitors.Monitor('dlp', width=config.SCREEN_WIDTH, distance=config.DISTANCE)
        win = visual.Window(fullscr = False, monitor=mon,
                        size = [_width,_height], viewScale = _viewScale,
                        pos = [_xpos,_ypos], screen = 1,
                        color=[-1,-1,-1],useFBO = True,allowGUI=False,
                        viewOri = 0.0)

        if exp_Info['WinMasks']: # Creating more than one screen to mask the main one
            win_mask_ls = window_3masks(win, _monitor=mon)

        # viewScale = [1,1/2] because dlp in patternMode has rectangular pixels
        # viewOri to compensate for the tilt of the projector.
        # If screen is already being tilted by Windows settings, set to 0.0 (deg)

    else: #In test mode on the PC screen
        _width,_height = 325, 325 # window size = 9cm in  my ASUS VG248 monitor
        _width,_height = 1920, 1080 # Full size in my ASUS VG248 monitor
        _width,_height = 1000, 1000 # window size = 18cm in  my Lenovo laptop
        _width,_height = 500, 500 # window size = 9cm in  my Lenovo laptop
        _width,_height = 512, 512 # window size = 9cm in  my Lenovo laptop + resizing the images for saving movie frames with imageio

        mon = monitors.Monitor('testMonitor', width=config.SCREEN_WIDTH, distance=config.DISTANCE)
        win = visual.Window(monitor=mon,size = [_width,_height], screen = 0,
                    allowGUI=False, color=[-1,-1,-1],useFBO = True, viewOri = 0.0)

        if exp_Info['WinMasks']: # Creating more than one screen to mask the main one
            win_mask_ls = window_3masks(win,_monitor=mon)

    # Gamma calibration
    if config.CALIBRATE_GAMMA:
        print(f'Psychopy gamma before calibration: {mon.getGamma()}')
        lum_measured = config.LUM_MEASURED
        lum_inputs = config.LUM_INPUTS
        gc = monitors.GammaCalculator(inputs=lum_inputs,lums=lum_measured,gamma=None, bitsIN=8, bitsOUT=8, eq=1)
        mon.setGamma(1/gc.gamma) #Inverse gamma to undo gamma calibration of the monitor
        print(f'Psychopy gamma after calibration: {mon.getGamma()}')
        #invFun = monitors.gammaInvFun(lum_measured, minLum = lum_measured[0], maxLum=lum_measured[-1], gamma=gc.gamma, b=None, eq=1)
        #gc.fitGammaFun(x=lum_inputs, y=lum_measured)
        #Gamma_1_8 = [0.0, 0.01347256, 0.04926537, 0.10517905, 0.18014963,0.27346917, 0.38461024, 0.51315452, 0.65875661, 0.82112322,1.0]

    # Other screen parameters (In the App, they are set in the "Monitor Center")
    #win.scrWidthCM = config.SCREEN_WIDTH # Width of the projection area of the screen
    #win.scrDistCM = config.DISTANCE # Distancefrom the viewer to the screen

    # #Detecting dropped frames if any
    # win.setRecordFrameIntervals(True)
    # # warn if frame is late more than 4 ms
    # win._refreshTreshold = 1/config.FRAMERATE+0.004
    # logging.console.setLevel(logging.WARNING)

##############################################################################
######################### Perspective correction #############################
##############################################################################

    # Subjects view perspective
    x_eyepoint = exp_Info['ViewPoint_x']
    y_eyepoint = exp_Info['ViewPoint_y']

    test_clock = core.Clock()
    #print(f'WARPER STARTS: {test_clock.getTime()+10}')
    # warp for perspective correction
    if stimdict["PERSPECTIVE_CORRECTION"]== 1:
        print('PERSPECTIVE CORRECTION APPLIED')
        warper = Warper(win, warp=exp_Info['Warp'],warpfile = "",
                    warpGridsize= 300, eyepoint = [x_eyepoint,y_eyepoint],
                    flipHorizontal = False, flipVertical = False)
        #warper.dist_cm = config.DISTANCE# debug_chris
        #warper.changeProjection(warp='spherical', eyepoint=(exp_Info['ViewPoint_x'], exp_Info['ViewPoint_y']))# debug_chris
        #print(f'Warper eyepoints: {warper.eyepoint}')
    else:
        warper = Warper(win, warp= None, eyepoint = [x_eyepoint,y_eyepoint])
    #print(f'WARPER ENDS: {test_clock.getTime()+10}')

##############################################################################
    #Printing screen info:
    print('##############################################')
    print('>>> Screen information:')
    print(f'Screen name: {mon.name}')
    print(f'Screen width: {config.SCREEN_WIDTH}')
    print(f'Distance to screen: {config.DISTANCE}')
    print(f'Covered visual angle: +- {round(max_angle_from_center(config.SCREEN_WIDTH, config.DISTANCE),2)}')
    print('##############################################')

##############################################################################

    # store frame rate of monitor if we can measure it
    exp_Info['actual_frameRate'] = win.getActualFrameRate()
    if exp_Info['Frame_rate'] != None:
        frameDur = 1.0 / round(exp_Info['Frame_rate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess

    # Forcing the MAXRUNTIME to be 0 in test mode
    if not dlp.OK:
        stimdict["MAXRUNTIME"] = 0


    # Write main setup to file (metadata)
    write_main_setup(config.OUT_DIR,dlp.OK,config.MAXRUNTIME,exp_Info)

    # shuffle epochs newly, if start or every epoch has been displayed
    if current_index == 0:
        try:
            print(f'RANDOMIZATION_MODE: {stimdict["RANDOMIZATION_MODE"]}')
            shuffle_index = shuffle_epochs(stimdict["RANDOMIZATION_MODE"],stimdict["EPOCHS"])
        except:
            shuffle_index = shuffle_epochs(stimdict["randomize"][0],stimdict["EPOCHS"]) # Seb, temp line for old stimuli design


##############################################################################
######### Creating some attributes per epoch (Stimulus object, bg, fg)########
##############################################################################
    # Generating or loading any stimulus data if STIMULUSDATA is not NULL
    if stimdict["STIMULUSDATA"] != "NULL":
            if stimdict["STIMULUSDATA"][0:10] == "SINUSOIDAL" or stimdict["STIMULUSDATA"][0:6] == "SQUARE":
                _useTex = True
                _useNoise = False
                # Creting texture for the sinusoidal grating
                dimension = 128 # It needs to be square power-of-two (e.g. 64 x 64) for PsychoPy
                if stimdict['stimtype'][-1] == 'noisy_circle':
                        dimension = config.FRAMERATE  # It needs to be the lentgh of the screen refresh (frame) rate for a proper frequency sampling

                stim_texture_ls = list()
                for e in range(stimdict["EPOCHS"]):
                    con= stimdict['michealson.contrast'][e]
                    lum = stimdict['lum'][e]
                    #Important here to recalculate bg and fg from original lum nand con values.
                    # The previous bg and fg were already corrected for dlp bit depth
                    # and [-1,1] color range. We want to avoud this here for BG and FG.
                    # Calculation of BG and FG here depends on the michelson contrast definition
                    FG = (con * lum) + lum #wrong: lum*(1+con)
                    BG = 2*lum - FG # wrong:lum*(1-con)
                    print(f'Sinusoidal wave, FG:{FG} BG:{BG}')
                    f = 1# generate a single cycle
                    # Generate 1D wave and modulate the luminance and contrast
                    x = np.arange(dimension)
                    # Wave needs to be scaled to 0-1 so we can modulate it easier later

                    if stimdict["STIMULUSDATA"][0:10] == "SINUSOIDAL":
                        sine_signal = (np.sin(2 * np.pi * f * x / dimension)/2 +0.5)

                        # Scaling the signal
                        #It stills need to me done differently. the MContrast scaling is not properly working and the scaling is not symmetric.
                        stim_texture  = (sine_signal  * 2*(FG - BG)* (63.0/255.0))-1 + (BG*(63.0/255.0)*2)# Scaling the signal to [-1,1] range, from 8bit to 6bit range and  to chosen MContrast

                    elif stimdict["STIMULUSDATA"][0:6] == "SQUARE":

                        FG = (1+ con)/2 # this calculation of contrast is valid for a constant mean luminance of 0.5, assuming michealson contrast
                        BG = 1-FG # this calculation of contrast is valid for a constant mean luminance of 0.5
                        x[0:64] = (FG * 2*(63.0/255.0)) -1
                        x[64:]= (BG * 2*(63.0/255.0)) -1
                        x=np.roll(x,-32)
                        stim_texture= x

                    stim_texture_min = np.min(stim_texture)

                    # Making either 1D or 2D sine wave
                    stim_texture = np.tile(stim_texture, [dimension,1]) # Saving 2D wave in the list
                    stim_texture_ls.append(stim_texture)



                if stimdict["STIMULUSDATA"][11:16] == "NOISY":
                    _useNoise = True
                    # Adding noise using target SNR
                    # Set a target SNR
                    # Creating noise array per epoch

                    # Max value of noise to avoid clipping of the sinusoidal wave
                    tolerated_noise_max_value_1 = 2*(abs(-1 - stim_texture_min)) # based on the lowest value for PSYCHOPY, -1
                    tolerated_noise_max_value_2 = np.min(stim_texture)-np.max(stim_texture) # based on the sinusoidal values. THIS CALCULATION ONLY MAKES SENSE FOR 50% MC

                    print('Noise levels (STD):')
                    noise_array_ls = list()
                    for i,SNR in enumerate(stimdict['SNR']):
                        target_snr = SNR
                        print(f'SNR {i}: {target_snr}')
                        # SNR as mean of standard deviation of signal/standard deviation of noise
                        # Wikipedia coeficient of variation definition

                        signal = stim_texture_ls[i][1]
                        signal_mean = np.mean(signal)
                        signal_std = np.std(signal)
                        signal_std = np.std(stim_texture_ls[i])
                        print(f'STD signal {i}: {signal_std}')
                        signal_rms = np.sqrt(np.mean(signal**2))
                        noise_mean = 0
                        noise_std = (signal_std/target_snr) # Before was: (signal_mean/target_snr)
                        print(f'STD {i}: {noise_std}')
                        noise_arr = np.random.normal(noise_mean, noise_std, [1000,dimension,dimension])
                        print(f'MAX VALUE {i}: {np.max(noise_arr)}')
                        if np.max(noise_arr) > tolerated_noise_max_value_1:
                            print(f'WARNING!!! NOISE CLIPPING FOR EPOCH: {i}')
                        noise_rms = np.sqrt(np.mean(noise_arr[0,:,:]**2))
                        noise_array_ls.append(noise_arr)

                        # Plotting what it will be presented
                        max_value = (2*(63.0/255.0))-1 # Max value in stim_texture after scaling
                        min_value = -1 # Min value in stim_texture after scaling
                        noisy_sinosoidal_wave = signal + noise_arr [1,1,:]
                        noisy_sinosoidal_wave[np.where(noisy_sinosoidal_wave> max_value)] = max_value
                        noisy_sinosoidal_wave[np.where(noisy_sinosoidal_wave<min_value)] = min_value
                        # plt.plot(noisy_sinosoidal_wave)
                        # plt.show()

                        # Calculating std for noise based on SNR definition in dB

                        # target_snr_db = 10* (np.log10(mean_signal/np.sqrt(noise_std)))
                        # target_snr_db = 10* (np.log10(signal_std/noise_std))
                        target_snr_db = 20* np.log10(signal_rms/noise_rms) # Wikipedia decibels definition
                        # print(f'SNR_dB {i}: {target_snr_db}')

                else:
                    noise_array_ls = list()
                    for e in range(stimdict["EPOCHS"]):
                        noise_array_ls.append(None)

            elif  stimdict["STIMULUSDATA"] == "TERNARY_TEXTURE":
                stim_texture_ls = list()
                noise_array_ls = list()
                choiseArr = [0,0.5,1]
                z= int(stimdict["texture.count"][1]) # 10000 # z- dimension (here frames presented over time)
                if int(stimdict["texture.hor_size"][1]) == 1:
                    x= 1 # x-dimension
                    y = int(stimdict["texture.vert_size"][1]) # y-dimension
                    np.random.seed(config.SEED)
                    stim_texture= np.random.choice(choiseArr, size=(z,x,y))
                    stim_texture = np.repeat(stim_texture,int(stimdict["texture.vert_size"][1]),axis=1)
                elif int(stimdict["texture.vert_size"][1]) == 1:
                    x= int(stimdict["texture.hor_size"][1])
                    y = 1
                    np.random.seed(config.SEED)
                    stim_texture= np.random.choice(choiseArr, size=(z,x,y))
                    stim_texture = np.repeat(stim_texture,int(stimdict["texture.hor_size"][1]),axis=2)
                else:
                    x=int(stimdict["texture.hor_size"][1])
                    y=int(stimdict["texture.vert_size"][1])
                    np.random.seed(config.SEED)
                    stim_texture= np.random.choice(choiseArr, size=(z,x,y))

                stim_texture_ls.append(stim_texture)
                noise_array_ls.append(None)


            elif stimdict["STIMULUSDATA"] =="Hi-res-noise":
                # super resolution approach (Pamlona et al. 2022)
                # Width of both dimensions have to be a divisor of screen dimensions, this will be based on the
                # rounded value of the screen angular dimensions since otherwise it is not easily possible 
                # to find a divisor
                ### generate random non repetitive array of integer for random shift in whitenoise stimulation (SR; Pamplona et al.)
                grayvalues=np.array([0,1.0,2.0]).astype('float64')
                grayvalues = np.where(grayvalues>-1,(grayvalues*63.0/255.0)-1,-1)
                #print(grayvalues)
                number_of_frames=15000
                stim_texture_ls = list()
                noise_array_ls = list()
                shifts = np.zeros((2, number_of_frames)) # 2 directions: x and y. 15000 random choices of shift that should be divisible by the shift resolution
                deg_topix='a'
                test = stimdict["Test"]


                ##test
                # shifts[0,:]=1
                # shifts[1,:]=0
                #end of test
                


                # set stimuli dimensions
                
                box_size_x = stimdict["Box_sizeX"] # first guess: this number should be around 20 
                box_size_y = stimdict["Box_sizeY"]

                frames=number_of_frames
                
                if 80%box_size_x == 0 and 80%box_size_y ==0:                
                            x_dim = int(80/box_size_x) #80 is the size of the screen in degrees
                            y_dim = int(80/box_size_y)
                else:
                    raise Exception ('in the current implementation, the code only accepts divisors of 80 as box_size')
                
                try:
                    np.random.seed(stimdict["seed"])
                    print(stimdict["seed"])
                except:
                    np.random.seed(3)
                    print(3)
                #test = stimdict["Test"]

                final_size=int(80/stimdict["Shift_resolution"])
                range_of_choice=range(int(-final_size/2), int(final_size/2))                
                                
                if stimdict["Shift_resolution"] != 0:
                    for i in range(0,2): # x,y shift arrays
                            np.random.seed(i)
                            shifts[i,:]= np.random.choice(range_of_choice, number_of_frames, replace=True)#range(int(np.floor(-38/stimdict["Shift_resolution"])),int(np.floor(40/stimdict["Shift_resolution"]))), number_of_frames, replace=True) #this range determines the x multiples of shifts 
                
                    #shifts=shifts*stimdict["Shift_resolution"]
                else:
                    pass
                print(shifts[0:5,:])
                print(np.max(shifts),np.min(shifts))

                # JF: in case we want the stimulus to remember the last shift
                #shifts=np.cumsum((shifts),axis=1)
                
                # if test:
                # noise_texture = np.random.choice(grayvalues, size=(1,y_dim,x_dim))
                # noise_texture = np.repeat(noise_texture,repeats=15000,axis=0)
                # else:
                try:
                    np.random.seed(stimdict["seed"])
                    print(stimdict["seed"])
                except:
                    np.random.seed(3)
                    print(3)
                noise_texture = np.random.choice(grayvalues, size=(number_of_frames,y_dim,x_dim))
                
                #upscale the stim array to be able to shift it in small

                if stimdict["Shift_resolution"] != 0:
                    #noise_texture = np.repeat(noise_texture,box_size_x,axis=1) # this brings the size of the matrix to the 80*80 size, then the shifts will be in the right scale
                    #noise_texture = np.repeat(noise_texture,box_size_y,axis=2)
                    upscale_factor_x = box_size_x/stimdict["Shift_resolution"]
                    upscale_factor_y = box_size_y/stimdict["Shift_resolution"]
                    noise_texture = np.repeat(noise_texture,upscale_factor_x,axis=1) # this brings the size of the matrix to the 80*80 size, then the shifts will be in the right scale
                    noise_texture = np.repeat(noise_texture,upscale_factor_y,axis=2)


                #copy_texture = copy.deepcopy(noise_texture)
                #print(np.unique(noise_texture))
                for frame in range(int(number_of_frames)):
                    
                    noise_texture[frame,:,:]=np.roll(noise_texture[frame,:,:], int(shifts[0,frame]),axis=0)
                    noise_texture[frame,:,:]=np.roll(noise_texture[frame,:,:], int(shifts[1,frame]),axis=1)
                    #test
                    #noise_texture[frame,:,:]=np.roll(noise_texture[frame,:,:], frame*1,axis=0)
                    #noise_texture[frame,:,:]=np.roll(noise_texture[frame,:,:], frame*0,axis=1)
                    ##end of test
                    

                if stimdict["print"] == 'True':
                    for frame in range(int(number_of_frames/100)):
                        plt.figure()
                        plt.imshow(noise_texture[frame,:,:],cmap='gray')
                        plt.savefig("C:\\#Coding\\pyVisualStim\\stimuli_collection\\7.High_resolution_WN\\pics\\_" + str(frame) + ".jpg")
                        plt.close()
                #     sys.exit()
                stim_texture_ls.append(noise_texture)
                
                if stimdict["print"] == 'True':

                    plt.figure()
                    plt.hist(shifts[0,:],bins=np.arange(-20,22,1))
                    plt.savefig("C:\\#Coding\\pyVisualStim\\stimuli_collection\\7.High_resolution_WN\\pics\\stimulus_hist%s_%sdegbox_%sdeg_shift.jpg"%(0,stimdict["Box_sizeX"],stimdict["Shift_resolution"]))
                    plt.close('all')
                    plt.figure()
                    plt.hist(shifts[1,:],bins=np.arange(-20,22,1))
                    plt.savefig("C:\\#Coding\\pyVisualStim\\stimuli_collection\\7.High_resolution_WN\\pics\\stimulus_hist%s_%sdegbox_%sdeg_shift.jpg"%(1,stimdict["Box_sizeX"],stimdict["Shift_resolution"]))
                    plt.close('all')

                    print(np.unique(stim_texture_ls))
                    print(np.max(np.array(stim_texture_ls)))
                    copy_texture = np.squeeze(np.array(stim_texture_ls)) # normalize the stimulus before saving)
                    print(copy_texture.shape)
                    print(np.unique(copy_texture))
                    np.save("C:\\#Coding\\pyVisualStim\\stimuli_collection\\7.High_resolution_WN\\pics\\stimulus_%sdegbox_%sdeg_shift.npy"%(stimdict["Box_sizeX"],stimdict["Shift_resolution"]),copy_texture)
                    

                    sys.exit()

            elif stimdict["STIMULUSDATA"] =="random_moving_binary_noise": 
                """ this is a stimulus intended to more strongly stimulate direction selective neurons.
                  it is similar to the hi_res aproach (pamplona et al, 2022) but shifts ocurr in predefined steps 
                  according to a movement speed
                  the steps of movement are decided randomly in each time step
                  
                  initially this stimulus is hardcoded to have binary luminance changes
                  
                  this stimulus type requires a speed of movement"""

                grayvalues = np.array([0,1]).astype('int8')
                #grayvalues = np.where(grayvalues>-1,(grayvalues*63.0/255.0)-1,-1)
                #print(grayvalues)
            
                if stimdict['type'] == 'Frozen':
                        number_of_frames = 4000
                        print('frozen type')
                else:
                        number_of_frames = 15000
                
                stim_texture_ls = list()
                noise_array_ls = list()
                moves = np.zeros((2, number_of_frames)) # 2 directions: x and y. 15000 random choices of shift that should be divisible by the shift resolution

                # set the resolution of the stimulus which is defined by the minimal movement posible, which is the lenght of movement given the speed and the framerate
                # ideally this should be an integer that should be able to divide without residue the final matrix size
                
                #step = int(float(stimdict['frame_duration'])*float(stimdict['speed']))

                step = 0.05*20 # maybe this should be hardcoded  ---- time resolution(s)*speed(deg/s)
                minimum_step = step*np.cos(np.deg2rad(45))
                allowed_box_sizes = [2,4,5,8,10,16] # more can be added

                # if 80%step != 0:
                #     raise Exception('in the current implementation, the code only accepts divisors of 80 as step')
                
                # set posible steps in units of stimulus pixels (the real step choice is -1*step,0,1*step)
                step_choices = [-1,0,1]
                               

                # set stimuli dimensions
                
                box_size_x = stimdict["Box_sizeX"] # before it was 5
                box_size_y = stimdict["Box_sizeY"] # before it was 5

                frames=number_of_frames
                
                x_dim = int(80//box_size_x) # 80 deg is an approximation of the dimension of the screen, this aproximation avoids the need for huge arrays

                # the precise dimension of the screen is (max_angle_from_center(9,5.3)*2)

                y_dim = int(80//box_size_y)

                minimum_size_step_based = int(80//step) #(max_angle_from_center(9,5.3)*2)
                minimum_size_diag_step = int(80//minimum_step)

                # temporal hardcoded DIAGONAL STEP to have an array size that reasonably approximates a diagonal step 
                # for speed 20deg and refresh period of 25ms  
                # at box size 5, speed 20 and refresh rate 0.05 minimum_size_diag_step 
                # can be approx to 120

                minimum_size_diag_step = 120 #this number has no physical meaning. it is an approximation to make the diagonal moves as fast as the lateral approx

                #minimum matrix dimensions that allow for aproxx equal speed in diagonal directions
                minimum_sizex=np.lcm.reduce([minimum_size_step_based,minimum_size_diag_step,x_dim]) 
                minimum_sizey=np.lcm.reduce([minimum_size_step_based,minimum_size_diag_step,y_dim])

                # define scaling factors
                upscale_factor_x = int(minimum_sizex/x_dim)
                upscale_factor_y = int(minimum_sizey/y_dim)

                resolutionx = 80/minimum_sizex
                resolutiony = 80/minimum_sizey
                step_multiplierx = int(step/resolutionx)
                step_multipliery = int(step/resolutiony)
                step_multiplier_diag = int(minimum_step/resolutionx)
                max_duration_scaler = int(float(stimdict['max_distance'])/resolutionx) # maximum movement posible(deg)/resolution. in this case max movement is stimdict['max_distance']
                                                      # this means the maximum duration of movement allows for up to stimdict['max_distance']. before it was hardcoded to 30
                min_duration_scaler = int(float(stimdict['min_distance'])/resolutionx) # before min_dist was hardcoded to 2 

                wrap_x = [-minimum_sizex//2 + 1 if minimum_sizex%2==0 else -minimum_sizex//2,  minimum_sizex//2]


                if eval(stimdict['persistent_movement']): # if we want the field to move consistently for a number of frames
                    # it takes tha same as the dimension number in steps to reach the original position
                   
                    choices_of_duration = np.array(range(min_duration_scaler,max_duration_scaler))# the maximum duration of a moving bout is determined by the maximum degree of movement allowed (26deg)

                    #temporal
                    #choices_of_duration = np.array([30,30])

                    if stimdict['type'] == 'Frozen':
                        persistant_val1,persistant_val2 = random_persistent_behavior_vector([8,9],number_of_frames,choices_of_duration) # function in helpers
                        moves1,moves2  =  random_persistent_values([persistant_val1,persistant_val2],[11,21],number_of_frames,step_choices,[1]) # function in helpers
                        print('pers_movement_frozen')
                    else:
                        persistant_val1,persistant_val2 = random_persistent_behavior_vector([4,5],number_of_frames,choices_of_duration) # function in helpers
                        moves1,moves2  =  random_persistent_values([persistant_val1,persistant_val2],[0,10],number_of_frames,step_choices,[1]) # function in helpers
                        print('pers_movement')

                    moves[0,:] = moves1
                    moves[1,:] = moves2
                    copy_moves = copy.deepcopy(moves)
                    #moves=np.cumsum((moves),axis=1).astype('int')
                    moves = helper_cumsum__wrap(moves,wrap_x[1],wrap_x[0])
                    #print('pers_movement')

                    # choose persistent luminances to enhance the motion signal relative to the luminance one

                else:

                    for i in range(0,2): # x,y shift arrays
                            np.random.seed(i)
                            moves[i,:]= np.random.choice(step_choices, number_of_frames, replace=True)#range(int(np.floor(-38/stimdict["Shift_resolution"])),int(np.floor(40/stimdict["Shift_resolution"]))), number_of_frames, replace=True) #this range determines the x multiples of shifts 

                    #moves=np.cumsum((moves),axis=1)                
                    moves = helper_cumsum__wrap(moves,wrap_x[1],wrap_x[0])

                #a=eval(stimdict['persistent_luminance'])

                if eval(stimdict['persistent_luminance'])==True:
                    #print('persistant lum')
                    if stimdict['type'] == 'Frozen':
                        print('persistant lum_Frozen')
                        persistant_lum = random_persistent_behavior_vector([7],number_of_frames,choices_of_duration)
                        noise_texture = random_persistent_values(persistant_lum,[15],number_of_frames,grayvalues,size=[x_dim,y_dim])
                        noise_texture = noise_texture[0].astype('int8')
                    else:

                        print('persistant lum')
                        persistant_lum = random_persistent_behavior_vector([5],number_of_frames,choices_of_duration)
                        noise_texture = random_persistent_values(persistant_lum,[3],number_of_frames,grayvalues,size=[x_dim,y_dim])
                        noise_texture = noise_texture[0].astype('int8')



                    #     # if test:
                    #     # noise_texture = np.random.choice(grayvalues, size=(1,y_dim,x_dim))
                    #     # noise_texture = np.repeat(noise_texture,repeats=15000,axis=0)
                    #     # else:
                else:
                    print('normal lum')
                    # try:
                    #     np.random.seed(stimdict["seed"])
                    #     print(stimdict["seed"])
                    # except:
                    np.random.seed(3)
                    # print(3)
                    noise_texture = np.random.choice(grayvalues, size=(number_of_frames,y_dim,x_dim))

                # scale luminance
                noise_texture = noise_texture*float(stimdict['lum_scaler'])
                print('lum:%s'%(float(stimdict['lum_scaler'])))

                #upscale the stim array to be able to shift it in small steps

                noise_texture = np.repeat(noise_texture,upscale_factor_x,axis=1) # this brings the size of the matrix to the 80*80 size, then the shifts will be in the right scale
                noise_texture = np.repeat(noise_texture,upscale_factor_y,axis=2)

                # scale the moves accordingly. diagonal movements require a different scaling
                for i in range(moves.shape[1]):
                    if int(np.abs(moves[0,i])) == 1 and int(np.abs(moves[1,i])) == 1:
                        moves[:,i] *= step_multiplier_diag
                    else:
                        moves[0,i] *= step_multiplierx
                        moves[1,i] *= step_multiplierx


                for frame in range(int(number_of_frames)):

                    noise_texture[frame,:,:] = np.roll(noise_texture[frame,:,:], int(moves[0,frame]),axis=0)
                    noise_texture[frame,:,:] = np.roll(noise_texture[frame,:,:], int(moves[1,frame]),axis=1)
                    #test
                    #noise_texture[frame,:,:]=np.roll(noise_texture[frame,:,:], frame*1,axis=0)
                    #noise_texture[frame,:,:]=np.roll(noise_texture[frame,:,:], frame*0,axis=1)
                    ##end of test


                # if stimdict["print"] == 'True':
                #     for frame in range(int(number_of_frames/100)):
                #         plt.figure()
                #         plt.imshow(noise_texture[frame,:,:],cmap='gray')
                #         plt.savefig("D:\\#Coding\\pyVisualStim\\stimuli_collection\\7.High_resolution_WN\\pics_movingnoise\\_" + str(frame) + ".jpg")
                #         plt.close()
                #     sys.exit()
                stim_texture_ls.append(noise_texture)

                if stimdict["print"] == 'True':

                    # plt.figure()
                    # plt.hist(moves[0,:],bins=np.arange(-20,22,1))
                    # plt.savefig("D:\\#Coding\\pyVisualStim\\stimuli_collection\\7.High_resolution_WN\\pics_movingnoise\\stimulus_hist%s_%sdegbox_%sdeg_step.jpg"%(0,stimdict["Box_sizeX"],step))
                    # plt.close('all')
                    # plt.figure()
                    # plt.hist(moves[1,:],bins=np.arange(-20,22,1))
                    # plt.savefig("D:\\#Coding\\pyVisualStim\\stimuli_collection\\7.High_resolution_WN\\pics_movingnoise\\stimulus_hist%s_%sdegbox_%sdeg_step.jpg"%(1,stimdict["Box_sizeX"],step))
                    # plt.close('all')

                    print(np.unique(stim_texture_ls))
                    print(np.max(np.array(stim_texture_ls)))
                    copy_texture = np.squeeze(np.array(stim_texture_ls)) # normalize the stimulus before saving)
                    print(copy_texture.shape)
                    print(np.unique(copy_texture))
                    #np.save("D:\\#Coding\\pyVisualStim\\stimuli_collection\\7.High_resolution_WN\\pics_movingnoise\\stimulus_%sdegbox_%sdeg_step.npy"%(stimdict["Box_sizeX"],step),copy_texture)
                    if stimdict["type"]=='Frozen':
                        copy_texture = copy_texture.astype('int8')
                        np.save("D:\\#Coding\\pyVisualStim\\stimuli_collection\\7.High_resolution_WN\\pics_movingnoise\\frozenstim_%smaxdur_%sdegbox_%sdeg_step.npy"%(int(float(stimdict['max_distance'])),stimdict["Box_sizeX"],step),copy_texture)
                        np.save("D:\\#Coding\\pyVisualStim\\stimuli_collection\\7.High_resolution_WN\\pics_movingnoise\\moves_frozenstim_%smaxdur_%sdegbox_%sdeg_step.npy"%(int(float(stimdict['max_distance'])),stimdict["Box_sizeX"],step),copy_moves)
                    else:
                        copy_texture = copy_texture.astype('int8')
                        np.save("D:\\#Coding\\pyVisualStim\\stimuli_collection\\7.High_resolution_WN\\pics_movingnoise\\stimulus_%smaxdur_%sdegbox_%sdeg_step.npy"%(int(float(stimdict['max_distance'])),stimdict["Box_sizeX"],step),copy_texture)
                        np.save("D:\\#Coding\\pyVisualStim\\stimuli_collection\\7.High_resolution_WN\\pics_movingnoise\\moves_%smaxdur_%sdegbox_%sdeg_step.npy"%(int(float(stimdict['max_distance'])),stimdict["Box_sizeX"],step),copy_moves)

                    sys.exit()

                if stimdict['type'] == 'Frozen':
                    stim_texture_ls = []
                    for e in range(stimdict["EPOCHS"]):
                        stim_texture_ls.append(noise_texture)

            elif stimdict["STIMULUSDATA"] =="random_moving_binary_manyspeeds_pluzfrozen": 
                """ this is a stimulus intended to more strongly stimulate direction selective neurons.
                  
                  initially this stimulus is hardcoded to have binary luminance changes
                  
                  this stimulus type implements multiple speeds of movement and includes a frozen sequence that repeats """

                grayvalues = np.array([0,1]).astype('int8')
                #grayvalues = np.where(grayvalues>-1,(grayvalues*63.0/255.0)-1,-1)
                #print(grayvalues)
                update_rate = 0.05 

                frozen_frames = int((1*60)/update_rate) # planned to last 1 minute with a refresh rate of 50ms
                #print( 'frozfr',frozen_frames)
                number_of_frames = int((12*60)/update_rate) # planned to last 12 minutes with a refresh rate of 50ms

                frozen_indices = [[0,1200],[3600,4800],[7200,8400],[10800,12000]] # in a 12 minute duration, put 4 repetitions of frozen noise

                stim_texture_ls = list()
                noise_array_ls = list()
                moves = np.zeros((2, number_of_frames)) # 2 directions: x and y. 15000 random choices of shift that should be divisible by the shift resolution

                # set the resolution of the stimulus which is defined by the minimal movement posible, which is the lenght of movement given the speed and the framerate
                # ideally this should be an integer that should be able to divide without residue the final matrix size

                #step = int(float(stimdict['frame_duration'])*float(stimdict['speed']))

                step = update_rate*20 # maybe this should be hardcoded  ---- time resolution(s)* minimal speed (this should allow for 0.5deg minimal movement
                               # but will increase in turn the size of the array passed. that could be risky)
                               # minimal speed is hardcoded at 10deg/s
                minimum_step = step*np.cos(np.deg2rad(45))

                allowed_box_sizes = [2,4,5,8,10,16] # more can be added


                step_choices = [-1,1,-2,0,2,-3,3] # adding more step choices adds more speed options: 10,20,40 deg/s

                # set stimuli dimensions

                box_size_x = stimdict["Box_sizeX"] 
                box_size_y = stimdict["Box_sizeY"] 

                frames=number_of_frames


                x_dim = int(80//box_size_x) # 80 deg is an approximation of the dimension of the screen, this aproximation avoids the need for huge arrays

                # the precise dimension of the screen is (max_angle_from_center(9,5.3)*2)

                y_dim = int(80//box_size_y)

                minimum_size_step_based = int(80//step) #(max_angle_from_center(9,5.3)*2)
                minimum_size_diag_step = int(80//minimum_step)

                # temporal hardcoded DIAGONAL STEP to have an array size that reasonably approximates a diagonal step 
                # for speed 20deg and refresh period of 25ms  
                # at box size 5, speed 20 and refresh rate 0.05 minimum_size_diag_step 
                # can be approx to 120

                minimum_size_diag_step = 120 #this number has no physical meaning. it is an approximation to make the diagonal moves as fast as the lateral approx

                #minimum matrix dimensions that allow for aproxx equal speed in diagonal directions
                minimum_sizex=np.lcm.reduce([minimum_size_step_based,minimum_size_diag_step,x_dim]) 
                minimum_sizey=np.lcm.reduce([minimum_size_step_based,minimum_size_diag_step,y_dim])

                # define scaling factors
                upscale_factor_x = int(minimum_sizex/x_dim)
                upscale_factor_y = int(minimum_sizey/y_dim)

                resolutionx = 80/minimum_sizex
                resolutiony = 80/minimum_sizey
                step_multiplierx = int(step/resolutionx)
                step_multipliery = int(step/resolutiony)
                step_multiplier_diag = int(minimum_step/resolutionx)
                max_duration_scaler = int(float(stimdict['max_duration'])/update_rate) # recommended 0.3s

                min_duration_scaler = int(float(stimdict['min_duration'])/update_rate) # recommended 0.05s # input here should be in seconds. when we divide by the stimulus refresh rate, then the result is in frames

                wrap_x = [-minimum_sizex//2 + 1 if minimum_sizex%2==0 else -minimum_sizex//2,  minimum_sizex//2]


                if eval(stimdict['persistent_movement']): # if we want the field to move consistently for a number of frames
                    # it takes tha same as the dimension number in steps to reach the original position

                    choices_of_duration = np.array(range(min_duration_scaler,max_duration_scaler))# the maximum duration of a moving bout is determined by the maximum degree of movement allowed (26deg)

                    persistant_val1,persistant_val2 = random_persistent_behavior_vector([4,5],number_of_frames,choices_of_duration) # function in helpers
                    moves1,moves2  =  random_persistent_values([persistant_val1,persistant_val2],[0,10],number_of_frames,step_choices,[1]) # function in helpers
                    print('pers_movement')
                    #print(moves1[0:20])

                    moves[0,:] = moves1
                    moves[1,:] = moves2
                    copy_moves = copy.deepcopy(moves)
                    #moves=np.cumsum((moves),axis=1).astype('int')
                    moves = helper_cumsum__wrap(moves,wrap_x[1],wrap_x[0])
                    #print('pers_movement')

                    # choose persistent luminances to enhance the motion signal relative to the luminance one
                else:

                    for i in range(0,2): # x,y shift arrays
                            np.random.seed(i)
                            moves[i,:]= np.random.choice(step_choices, number_of_frames, replace=True)#range(int(np.floor(-38/stimdict["Shift_resolution"])),int(np.floor(40/stimdict["Shift_resolution"]))), number_of_frames, replace=True) #this range determines the x multiples of shifts 

                    #moves=np.cumsum((moves),axis=1)                
                    moves = helper_cumsum__wrap(moves,wrap_x[1],wrap_x[0])

                #a=eval(stimdict['persistent_luminance'])

                if eval(stimdict['persistent_luminance'])==True:
                    #print('persistant lum')
                    print('persistant lum')
                    persistant_lum = random_persistent_behavior_vector([5],number_of_frames,choices_of_duration)
                    noise_texture = random_persistent_values(persistant_lum,[3],number_of_frames,grayvalues,size=[x_dim,y_dim])
                    noise_texture = noise_texture[0].astype('int8')

                    #     # if test:
                    #     # noise_texture = np.random.choice(grayvalues, size=(1,y_dim,x_dim))
                    #     # noise_texture = np.repeat(noise_texture,repeats=15000,axis=0)
                    #     # else:
                else:
                    print('normal lum')
                    # try:
                    #     np.random.seed(stimdict["seed"])
                    #     print(stimdict["seed"])
                    # except:
                    np.random.seed(3)
                    # print(3)
                    noise_texture = np.random.choice(grayvalues, size=(number_of_frames,y_dim,x_dim))

                # scale luminance
                noise_texture = noise_texture*float(stimdict['lum_scaler'])
                print('lum:%s'%(float(stimdict['lum_scaler'])))

                ###################
                #### introduce size changes
                ###################

                if eval(stimdict['multiple_sizes']):
                    duration_choices = np.array(range(min_duration_scaler,max_duration_scaler))
                    posible_sizes = eval(stimdict['pos_sizes'])#[1,2,4] # this should be understood as value times x the original size. if original size is 
                    sizes_1, sizes_2 = random_persistent_behavior_vector([500,600],number_of_frames,duration_choices)
                    persistent_sizes_1, persistent_sizes_2 = random_persistent_values([sizes_1, sizes_2],[1050,2050],number_of_frames,posible_sizes,[1])
                    persistent_sizes_1 = persistent_sizes_1.astype('int')
                    persistent_sizes_2 = persistent_sizes_2.astype('int')

                    for time_ix in range(noise_texture.shape[0]):
                        # (full size - dim_size)/2
                        #temporal_texture = np.kron(noise_texture[time_ix,:,:], np.ones((persistent_sizes_1[time_ix],persistent_sizes_2[time_ix])))
                        temporal_texture = np.repeat(noise_texture[time_ix,:,:],persistent_sizes_1[time_ix],axis=0)
                        temporal_texture = np.repeat(temporal_texture,persistent_sizes_2[time_ix],axis=1)
                        x_slice = int((temporal_texture.shape[0]-x_dim)/2)
                        y_slice = int((temporal_texture.shape[1]-y_dim)/2)                       
                        noise_texture[time_ix,:,:] = temporal_texture[x_slice : x_slice + x_dim,y_slice : y_slice + y_dim]

                ####################################
                # add_frozen part
                ####################################

                f_moves = np.zeros((2, frozen_frames))

                ####################################
                # add_frozen part --> movements
                ####################################

                persistant_val1,persistant_val2 = random_persistent_behavior_vector([8,9],frozen_frames,choices_of_duration) # function in helpers
                f_moves1,f_moves2  =  random_persistent_values([persistant_val1,persistant_val2],[11,21],frozen_frames,step_choices,[1]) # function in helpers
                print('pers_movement_frozen')

                f_moves[0,:] = f_moves1
                f_moves[1,:] = f_moves2
                copy_moves = copy.deepcopy(f_moves)
                #moves=np.cumsum((moves),axis=1).astype('int')
                f_moves = helper_cumsum__wrap(f_moves,wrap_x[1],wrap_x[0])



                #######introduce the moves in the bigger moves array

                for interval in frozen_indices:
                    #print('int',interval)
                    moves[:,interval[0]:interval[1]] = f_moves
                ###########
                ############# create frozen luminances
                ###########
                if eval(stimdict['persistent_luminance'])==True:
                    frozen_lum = random_persistent_behavior_vector([7],frozen_frames,choices_of_duration)
                    frozen = random_persistent_values(frozen_lum,[15],frozen_frames,grayvalues,size=[x_dim,y_dim])
                    frozen = frozen[0].astype('int8')
                else:
                    np.random.seed(120)
                    # print(3)
                    frozen = np.random.choice(grayvalues, size=(frozen_frames,y_dim,x_dim))

                ######################
                ########### introduce frozen sizes
                ###################

                if eval(stimdict['multiple_sizes']):
                    duration_choices = np.array(range(min_duration_scaler,max_duration_scaler)) # the duration of a size at the minimum is a single frame, and at a maximum is user defined
                    #posible_sizes = [1,2,4] # this should be understood as value times x the original size. if original size is 
                    sizes_1, sizes_2 = random_persistent_behavior_vector([250,300],frozen_frames,duration_choices)
                    persistent_sizes_1, persistent_sizes_2 = random_persistent_values([sizes_1, sizes_2],[600,800],frozen_frames,posible_sizes,[1])
                    persistent_sizes_1 = persistent_sizes_1.astype('int')
                    persistent_sizes_2 = persistent_sizes_2.astype('int')
                    for time_ix in range(frozen_frames):
                        # (full size - dim_size)/2 
                        #temporal_texture = np.kron(noise_texture[time_ix,:,:], np.ones((persistent_sizes_1[time_ix],persistent_sizes_2[time_ix])))

                        temporal_texture = np.repeat(frozen[time_ix,:,:],persistent_sizes_1[time_ix],axis=0)
                        temporal_texture = np.repeat(temporal_texture,persistent_sizes_2[time_ix],axis=1)

                        x_slice = int((temporal_texture.shape[0]-x_dim)/2)
                        y_slice = int((temporal_texture.shape[1]-y_dim)/2)
                        frozen[time_ix,:,:] = temporal_texture[x_slice : x_slice + x_dim,y_slice : y_slice + y_dim]
                print('worked')
                ############ introduce frozen luminances. but first scale the luminance

                frozen = frozen*float(stimdict['lum_scaler'])
                #print('lum:%s'%(float(stimdict['lum_scaler'])))

                for interval in frozen_indices:
                    noise_texture[interval[0]:interval[1],:,:] = frozen

                #upscale the stim array to be able to shift it in small steps

                noise_texture = np.repeat(noise_texture,upscale_factor_x,axis=1) # this brings the size of the matrix to the 80*80 size, then the shifts will be in the right scale
                noise_texture = np.repeat(noise_texture,upscale_factor_y,axis=2)

                # scale the moves accordingly. diagonal movements require a different scaling
                for i in range(moves.shape[1]):
                    if int(np.abs(moves[0,i])) != 0 and int(np.abs(moves[1,i])) != 0:
                        moves[:,i] *= step_multiplier_diag

                    else:
                        moves[0,i] *= step_multiplierx
                        moves[1,i] *= step_multiplierx



                for frame in range(int(number_of_frames)):

                    noise_texture[frame,:,:] = np.roll(noise_texture[frame,:,:], int(moves[0,frame]),axis=0)





                    noise_texture[frame,:,:] = np.roll(noise_texture[frame,:,:], int(moves[1,frame]),axis=1)
                    #test
                    #noise_texture[frame,:,:]=np.roll(noise_texture[frame,:,:], frame*1,axis=0)
                    #noise_texture[frame,:,:]=np.roll(noise_texture[frame,:,:], frame*0,axis=1)
                    ##end of test


                # if stimdict["print"] == 'True':
                #     for frame in range(int(number_of_frames/100)):
                #         plt.figure()
                #         plt.imshow(noise_texture[frame,:,:],cmap='gray')
                #         plt.savefig("D:\\#Coding\\pyVisualStim\\stimuli_collection\\7.High_resolution_WN\\pics_movingnoise\\_" + str(frame) + ".jpg")
                #         plt.close()
                #     sys.exit()
                stim_texture_ls.append(noise_texture)

                if stimdict["print"] == 'True':

                    # plt.figure()
                    # plt.hist(moves[0,:],bins=np.arange(-20,22,1))
                    # plt.savefig("D:\\#Coding\\pyVisualStim\\stimuli_collection\\7.High_resolution_WN\\pics_movingnoise\\stimulus_hist%s_%sdegbox_%sdeg_step.jpg"%(0,stimdict["Box_sizeX"],step))
                    # plt.close('all')
                    # plt.figure()
                    # plt.hist(moves[1,:],bins=np.arange(-20,22,1))
                    # plt.savefig("D:\\#Coding\\pyVisualStim\\stimuli_collection\\7.High_resolution_WN\\pics_movingnoise\\stimulus_hist%s_%sdegbox_%sdeg_step.jpg"%(1,stimdict["Box_sizeX"],step))
                    # plt.close('all')

                    print(np.unique(stim_texture_ls))
                    print(np.max(np.array(stim_texture_ls)))
                    copy_texture = np.squeeze(np.array(stim_texture_ls)) # normalize the stimulus before saving)
                    print(copy_texture.shape)
                    print(np.unique(copy_texture))
                    #np.save("D:\\#Coding\\pyVisualStim\\stimuli_collection\\7.High_resolution_WN\\pics_movingnoise\\stimulus_%sdegbox_%sdeg_step.npy"%(stimdict["Box_sizeX"],step),copy_texture)
                    if stimdict["type"]=='Frozen':
                        copy_texture = copy_texture.astype('int8')
                        np.save("D:\\#Coding\\pyVisualStim\\stimuli_collection\\7.High_resolution_WN\\pics_movingnoise\\frozenstim_%smaxdur_%sdegbox_%sdeg_step.npy"%(int(float(stimdict['max_distance'])),stimdict["Box_sizeX"],step),copy_texture)
                        np.save("D:\\#Coding\\pyVisualStim\\stimuli_collection\\7.High_resolution_WN\\pics_movingnoise\\moves_frozenstim_%smaxdur_%sdegbox_%sdeg_step.npy"%(int(float(stimdict['max_distance'])),stimdict["Box_sizeX"],step),copy_moves)
                    else:
                        copy_texture = copy_texture.astype('int8')
                        np.save("D:\\#Coding\\pyVisualStim\\stimuli_collection\\7.High_resolution_WN\\pics_movingnoise\\stimulus_%smaxdur_%sdegbox_%sdeg_step.npy"%(int(float(stimdict['max_distance'])),stimdict["Box_sizeX"],step),copy_texture)
                        np.save("D:\\#Coding\\pyVisualStim\\stimuli_collection\\7.High_resolution_WN\\pics_movingnoise\\moves_%smaxdur_%sdegbox_%sdeg_step.npy"%(int(float(stimdict['max_distance'])),stimdict["Box_sizeX"],step),copy_moves)

                    sys.exit()

                if stimdict['type'] == 'Frozen':
                    stim_texture_ls = []
                    for e in range(stimdict["EPOCHS"]):
                        stim_texture_ls.append(noise_texture)

            elif stimdict["STIMULUSDATA"] =="random_moving_binary_manyspeeds_pluzfrozen": 
                """ this is a stimulus intended to more strongly stimulate direction selective neurons.
                  
                  initially this stimulus is hardcoded to have binary luminance changes
                  
                  this stimulus type implements multiple speeds of movement and includes a frozen sequence that repeats """

                grayvalues = np.array([0,1]).astype('int8')
                #grayvalues = np.where(grayvalues>-1,(grayvalues*63.0/255.0)-1,-1)
                #print(grayvalues)
                update_rate = 0.05 

                frozen_frames = int((1*60)/update_rate) # planned to last 1 minute with a refresh rate of 50ms
                #print( 'frozfr',frozen_frames)
                number_of_frames = int((12*60)/update_rate) # planned to last 12 minutes with a refresh rate of 50ms

                frozen_indices = [[0,1200],[3600,4800],[7200,8400],[10800,12000]] # in a 12 minute duration, put 4 repetitions of frozen noise

                stim_texture_ls = list()
                noise_array_ls = list()
                moves = np.zeros((2, number_of_frames)) # 2 directions: x and y. 15000 random choices of shift that should be divisible by the shift resolution

                # set the resolution of the stimulus which is defined by the minimal movement posible, which is the lenght of movement given the speed and the framerate
                # ideally this should be an integer that should be able to divide without residue the final matrix size

                #step = int(float(stimdict['frame_duration'])*float(stimdict['speed']))

                step = update_rate*20 # maybe this should be hardcoded  ---- time resolution(s)* minimal speed (this should allow for 0.5deg minimal movement
                               # but will increase in turn the size of the array passed. that could be risky)
                               # minimal speed is hardcoded at 10deg/s
                minimum_step = step*np.cos(np.deg2rad(45))

                allowed_box_sizes = [2,4,5,8,10,16] # more can be added


                step_choices = [-1,1,-2,0,2,-3,3] # adding more step choices adds more speed options: 10,20,40 deg/s

                # set stimuli dimensions

                box_size_x = stimdict["Box_sizeX"] 
                box_size_y = stimdict["Box_sizeY"] 

                frames=number_of_frames


                x_dim = int(80//box_size_x) # 80 deg is an approximation of the dimension of the screen, this aproximation avoids the need for huge arrays

                # the precise dimension of the screen is (max_angle_from_center(9,5.3)*2)

                y_dim = int(80//box_size_y)

                minimum_size_step_based = int(80//step) #(max_angle_from_center(9,5.3)*2)
                minimum_size_diag_step = int(80//minimum_step)

                # temporal hardcoded DIAGONAL STEP to have an array size that reasonably approximates a diagonal step 
                # for speed 20deg and refresh period of 25ms  
                # at box size 5, speed 20 and refresh rate 0.05 minimum_size_diag_step 
                # can be approx to 120

                minimum_size_diag_step = 120 #this number has no physical meaning. it is an approximation to make the diagonal moves as fast as the lateral approx

                #minimum matrix dimensions that allow for aproxx equal speed in diagonal directions
                minimum_sizex=np.lcm.reduce([minimum_size_step_based,minimum_size_diag_step,x_dim]) 
                minimum_sizey=np.lcm.reduce([minimum_size_step_based,minimum_size_diag_step,y_dim])

                # define scaling factors
                upscale_factor_x = int(minimum_sizex/x_dim)
                upscale_factor_y = int(minimum_sizey/y_dim)

                resolutionx = 80/minimum_sizex
                resolutiony = 80/minimum_sizey
                step_multiplierx = int(step/resolutionx)
                step_multipliery = int(step/resolutiony)
                step_multiplier_diag = int(minimum_step/resolutionx)
                max_duration_scaler = int(float(stimdict['max_duration'])/update_rate) # recommended 0.3s

                min_duration_scaler = int(float(stimdict['min_duration'])/update_rate) # recommended 0.05s # input here should be in seconds. when we divide by the stimulus refresh rate, then the result is in frames

                wrap_x = [-minimum_sizex//2 + 1 if minimum_sizex%2==0 else -minimum_sizex//2,  minimum_sizex//2]


                if eval(stimdict['persistent_movement']): # if we want the field to move consistently for a number of frames
                    # it takes tha same as the dimension number in steps to reach the original position

                    choices_of_duration = np.array(range(min_duration_scaler,max_duration_scaler))# the maximum duration of a moving bout is determined by the maximum degree of movement allowed (26deg)

                    persistant_val1,persistant_val2 = random_persistent_behavior_vector([4,5],number_of_frames,choices_of_duration) # function in helpers
                    moves1,moves2  =  random_persistent_values([persistant_val1,persistant_val2],[0,10],number_of_frames,step_choices,[1]) # function in helpers
                    print('pers_movement')
                    #print(moves1[0:20])

                    moves[0,:] = moves1
                    moves[1,:] = moves2
                    copy_moves = copy.deepcopy(moves)
                    #moves=np.cumsum((moves),axis=1).astype('int')
                    moves = helper_cumsum__wrap(moves,wrap_x[1],wrap_x[0])
                    #print('pers_movement')

                    # choose persistent luminances to enhance the motion signal relative to the luminance one
                else:

                    for i in range(0,2): # x,y shift arrays
                            np.random.seed(i)
                            moves[i,:]= np.random.choice(step_choices, number_of_frames, replace=True)#range(int(np.floor(-38/stimdict["Shift_resolution"])),int(np.floor(40/stimdict["Shift_resolution"]))), number_of_frames, replace=True) #this range determines the x multiples of shifts 

                    #moves=np.cumsum((moves),axis=1)                
                    moves = helper_cumsum__wrap(moves,wrap_x[1],wrap_x[0])

                #a=eval(stimdict['persistent_luminance'])

                if eval(stimdict['persistent_luminance'])==True:
                    #print('persistant lum')
                    print('persistant lum')
                    persistant_lum = random_persistent_behavior_vector([5],number_of_frames,choices_of_duration)
                    noise_texture = random_persistent_values(persistant_lum,[3],number_of_frames,grayvalues,size=[x_dim,y_dim])
                    noise_texture = noise_texture[0].astype('int8')

                    #     # if test:
                    #     # noise_texture = np.random.choice(grayvalues, size=(1,y_dim,x_dim))
                    #     # noise_texture = np.repeat(noise_texture,repeats=15000,axis=0)
                    #     # else:
                else:
                    print('normal lum')
                    # try:
                    #     np.random.seed(stimdict["seed"])
                    #     print(stimdict["seed"])
                    # except:
                    np.random.seed(3)
                    # print(3)
                    noise_texture = np.random.choice(grayvalues, size=(number_of_frames,y_dim,x_dim))

                # scale luminance
                noise_texture = noise_texture*float(stimdict['lum_scaler'])
                print('lum:%s'%(float(stimdict['lum_scaler'])))

                ###################
                #### introduce size changes
                ###################

                if eval(stimdict['multiple_sizes']):
                    duration_choices = np.array(range(min_duration_scaler,max_duration_scaler))
                    posible_sizes = eval(stimdict['pos_sizes'])#[1,2,4] # this should be understood as value times x the original size. if original size is 
                    sizes_1, sizes_2 = random_persistent_behavior_vector([500,600],number_of_frames,duration_choices)
                    persistent_sizes_1, persistent_sizes_2 = random_persistent_values([sizes_1, sizes_2],[1050,2050],number_of_frames,posible_sizes,[1])
                    persistent_sizes_1 = persistent_sizes_1.astype('int')
                    persistent_sizes_2 = persistent_sizes_2.astype('int')

                    for time_ix in range(noise_texture.shape[0]):
                        # (full size - dim_size)/2
                        #temporal_texture = np.kron(noise_texture[time_ix,:,:], np.ones((persistent_sizes_1[time_ix],persistent_sizes_2[time_ix])))
                        temporal_texture = np.repeat(noise_texture[time_ix,:,:],persistent_sizes_1[time_ix],axis=0)
                        temporal_texture = np.repeat(temporal_texture,persistent_sizes_2[time_ix],axis=1)
                        x_slice = int((temporal_texture.shape[0]-x_dim)/2)
                        y_slice = int((temporal_texture.shape[1]-y_dim)/2)                       
                        noise_texture[time_ix,:,:] = temporal_texture[x_slice : x_slice + x_dim,y_slice : y_slice + y_dim]

                ####################################
                # add_frozen part
                ####################################

                f_moves = np.zeros((2, frozen_frames))

                ####################################
                # add_frozen part --> movements
                ####################################

                persistant_val1,persistant_val2 = random_persistent_behavior_vector([8,9],frozen_frames,choices_of_duration) # function in helpers
                f_moves1,f_moves2  =  random_persistent_values([persistant_val1,persistant_val2],[11,21],frozen_frames,step_choices,[1]) # function in helpers
                print('pers_movement_frozen')

                f_moves[0,:] = f_moves1
                f_moves[1,:] = f_moves2
                copy_moves = copy.deepcopy(f_moves)
                #moves=np.cumsum((moves),axis=1).astype('int')
                f_moves = helper_cumsum__wrap(f_moves,wrap_x[1],wrap_x[0])



                #######introduce the moves in the bigger moves array

                for interval in frozen_indices:
                    #print('int',interval)
                    moves[:,interval[0]:interval[1]] = f_moves
                ###########
                ############# create frozen luminances
                ###########
                if eval(stimdict['persistent_luminance'])==True:
                    frozen_lum = random_persistent_behavior_vector([7],frozen_frames,choices_of_duration)
                    frozen = random_persistent_values(frozen_lum,[15],frozen_frames,grayvalues,size=[x_dim,y_dim])
                    frozen = frozen[0].astype('int8')
                else:
                    np.random.seed(120)
                    # print(3)
                    frozen = np.random.choice(grayvalues, size=(frozen_frames,y_dim,x_dim))

                ######################
                ########### introduce frozen sizes
                ###################

                if eval(stimdict['multiple_sizes']):
                    duration_choices = np.array(range(min_duration_scaler,max_duration_scaler)) # the duration of a size at the minimum is a single frame, and at a maximum is user defined
                    #posible_sizes = [1,2,4] # this should be understood as value times x the original size. if original size is 
                    sizes_1, sizes_2 = random_persistent_behavior_vector([250,300],frozen_frames,duration_choices)
                    persistent_sizes_1, persistent_sizes_2 = random_persistent_values([sizes_1, sizes_2],[600,800],frozen_frames,posible_sizes,[1])
                    persistent_sizes_1 = persistent_sizes_1.astype('int')
                    persistent_sizes_2 = persistent_sizes_2.astype('int')
                    for time_ix in range(frozen_frames):
                        # (full size - dim_size)/2 
                        #temporal_texture = np.kron(noise_texture[time_ix,:,:], np.ones((persistent_sizes_1[time_ix],persistent_sizes_2[time_ix])))

                        temporal_texture = np.repeat(frozen[time_ix,:,:],persistent_sizes_1[time_ix],axis=0)
                        temporal_texture = np.repeat(temporal_texture,persistent_sizes_2[time_ix],axis=1)

                        x_slice = int((temporal_texture.shape[0]-x_dim)/2)
                        y_slice = int((temporal_texture.shape[1]-y_dim)/2)
                        frozen[time_ix,:,:] = temporal_texture[x_slice : x_slice + x_dim,y_slice : y_slice + y_dim]
                print('worked')
                ############ introduce frozen luminances. but first scale the luminance

                frozen = frozen*float(stimdict['lum_scaler'])
                #print('lum:%s'%(float(stimdict['lum_scaler'])))

                for interval in frozen_indices:
                    noise_texture[interval[0]:interval[1],:,:] = frozen

                #upscale the stim array to be able to shift it in small steps

                noise_texture = np.repeat(noise_texture,upscale_factor_x,axis=1) # this brings the size of the matrix to the 80*80 size, then the shifts will be in the right scale
                noise_texture = np.repeat(noise_texture,upscale_factor_y,axis=2)

                # scale the moves accordingly. diagonal movements require a different scaling
                for i in range(moves.shape[1]):
                    if int(np.abs(moves[0,i])) != 0 and int(np.abs(moves[1,i])) != 0:
                        moves[:,i] *= step_multiplier_diag

                    else:
                        moves[0,i] *= step_multiplierx
                        moves[1,i] *= step_multiplierx



                for frame in range(int(number_of_frames)):

                    noise_texture[frame,:,:] = np.roll(noise_texture[frame,:,:], int(moves[0,frame]),axis=0)
                    noise_texture[frame,:,:] = np.roll(noise_texture[frame,:,:], int(moves[1,frame]),axis=1)
                    #test
                    #noise_texture[frame,:,:]=np.roll(noise_texture[frame,:,:], frame*1,axis=0)
                    #noise_texture[frame,:,:]=np.roll(noise_texture[frame,:,:], frame*0,axis=1)
                    ##end of test
                    

                # if stimdict["print"] == 'True':
                #     for frame in range(int(number_of_frames/100)):
                #         plt.figure()
                #         plt.imshow(noise_texture[frame,:,:],cmap='gray')
                #         plt.savefig("D:\\#Coding\\pyVisualStim\\stimuli_collection\\7.High_resolution_WN\\pics_movingnoise\\_" + str(frame) + ".jpg")
                #         plt.close()
                #     sys.exit()
                stim_texture_ls.append(noise_texture)

                if stimdict["print"] == 'True':

                    # plt.figure()
                    # plt.hist(moves[0,:],bins=np.arange(-20,22,1))
                    # plt.savefig("D:\\#Coding\\pyVisualStim\\stimuli_collection\\7.High_resolution_WN\\pics_movingnoise\\stimulus_hist%s_%sdegbox_%sdeg_step.jpg"%(0,stimdict["Box_sizeX"],step))
                    # plt.close('all')
                    # plt.figure()
                    # plt.hist(moves[1,:],bins=np.arange(-20,22,1))
                    # plt.savefig("D:\\#Coding\\pyVisualStim\\stimuli_collection\\7.High_resolution_WN\\pics_movingnoise\\stimulus_hist%s_%sdegbox_%sdeg_step.jpg"%(1,stimdict["Box_sizeX"],step))
                    # plt.close('all')

                    print(np.unique(stim_texture_ls))
                    print(np.max(np.array(stim_texture_ls)))
                    copy_texture = np.squeeze(np.array(stim_texture_ls)) # normalize the stimulus before saving)
                    print(copy_texture.shape)
                    print(np.unique(copy_texture))
                    #np.save("D:\\#Coding\\pyVisualStim\\stimuli_collection\\7.High_resolution_WN\\pics_movingnoise\\stimulus_%sdegbox_%sdeg_step.npy"%(stimdict["Box_sizeX"],step),copy_texture)

                    copy_texture = copy_texture.astype('int8')
                    np.save("D:\\#Coding\\pyVisualStim\\stimuli_collection\\7.High_resolution_WN\\pics_movingnoise_manyspeeds_frozen\\stimulus_%smaxdur_%sdegbox_%sdeg_step.npy"%(int(float(stimdict['max_duration'])),stimdict["Box_sizeX"],step),copy_texture)
                    np.save("D:\\#Coding\\pyVisualStim\\stimuli_collection\\7.High_resolution_WN\\pics_movingnoise_manyspeeds_frozen\\moves_%smaxdur_%sdegbox_%sdeg_step.npy"%(int(float(stimdict['max_duration'])),stimdict["Box_sizeX"],step),copy_moves)

                    sys.exit()

            elif  stimdict["STIMULUSDATA"] == "POLIGON":
                stim_texture_ls = list()
                noise_array_ls = list()
                x=int(stimdict["texture.hor_size"][1])
                y=int(stimdict["texture.vert_size"][1])
                z= 10000 # z- dimension (here frames presented over time)
                curr_arr = np.zeros(size=(z,x,y))

            elif stimdict["STIMULUSDATA"] == "Gaussian":  # Pradeep
                stim_texture_ls = list()
                noise_array_ls = list()
                for e in range(stimdict["EPOCHS"]):
                    stim_texture_ls.append(None)
                    noise_array_ls.append(None)
                _useTex = False
                _useNoise = False
                lum_gaussian = guassian_distributed_lums(seed = 50, lum_vector_size= 2500) 



            else: # Specific case for older files (used in 2pstim-C- in which ["STIMULUSDATA"] was not specified
                 stim_texture = h5py.File(stimdict["STIMULUSDATA"])
                 stim_texture= stim_texture['stimulus'][()]
                 stim_texture= stim_texture[0:10000,:,:] # 10000 is a fix value

    else: # When ["STIMULUSDATA"]is == "NULL"
        stim_texture_ls = list()
        noise_array_ls = list()
        for e in range(stimdict["EPOCHS"]):
            stim_texture_ls.append(None)
            noise_array_ls.append(None)
        _useTex = False
        _useNoise = False

    # Creating the stimulus object per epoch
    stim_object_ls = list()
    for i,stimtype in enumerate(stimdict["stimtype"]):
        if stimdict["PERSPECTIVE_CORRECTION"] == 1:
            _units = 'deg' # Keep in "deg" when using the warper.

        else:
            #'degFlatPos' is the correct unit for having a correct screen size
            # in degrees when the perspective is not corrected by the warper.
            _units = 'degFlatPos'

        if stimtype[-1] == "C":
            circle = visual.Circle(win, units=_units, edges = 128)
            stim_object = circle
        
        elif stimtype == "GaussCircle":      # Pradeep
            circle = visual.Circle(win, units=_units, edges = 128)
            stim_object = circle

        elif stimtype[-1] == "NC":
            circle = visual.Circle(win, units=_units, edges = 128)
            stim_object = circle

        elif stimtype ==  "SSR":
            bar = visual.Rect(win, lineWidth=0, units=_units)
            stim_object = bar

        elif stimtype ==  "R":
            bar = visual.Rect(win, lineWidth=0, units=_units)
            stim_object = bar

        elif stimtype ==  "DS":
            bar = visual.Rect(win, lineWidth=0, units=_units)
            stim_object = bar
            
        elif stimtype ==  "ADS":
            bar = visual.Rect(win, lineWidth=0, units=_units)
            stim_object = bar

        elif stimtype ==  "RDS":
            bar = visual.Rect(win, lineWidth=0, units=_units)
            stim_object = bar
        
        elif stimtype == "N":
            noise = visual.GratingStim(win,units=_units, name='noise',tex='sqr')
            stim_object = noise

        elif stimtype=="shift_noise":
            noise = visual.GratingStim(win,units=_units, name='noise',tex='sqr')#,pos=stim_pos,size=stim_size)
            stim_object = noise

        elif stimtype[-1:] == "G":
            grating = visual.GratingStim(win,units=_units, name='grating',
                                         tex='sqr',colorSpace='rgb',
                                         blendmode='avg',texRes=128,
                                         interpolate=True, depth=-1.0,
                                         phase = (0,0))
            # noise = visual.NoiseStim(win,units=_units, name='noise',
            #                          colorSpace='rgb',noiseType='Binary',
            #                          noiseElementSize=0.0625,noiseBaseSf=8.0,
            #                          noiseBW=1,noiseBWO=30, noiseOri=0.0,
            #                          noiseFractalPower=0.0,noiseFilterLower=1.0,
            #                          noiseFilterUpper=8.0, noiseFilterOrder=0.0,
            #                          noiseClip=3.0, interpolate=False, depth=0.0)
            # noise.buildNoise()
            stim_object =grating

        elif stimtype ==  "DG":
            grating = visual.GratingStim(win,units=_units, name='grating',
                                         tex='sqr',colorSpace='rgb',blendmode='avg',
                                         texRes=128, interpolate=True, depth=-1.0,
                                         phase = (0,0))
            dots = visual.DotStim( win=win, name='dots', units=_units,
                                  nDots=int(stimdict["nDots"][i]), dotSize=5,
                                  speed=0.1, dir=0.0, coherence=1.0,
                                  fieldPos=(0.0, 0.0), fieldSize=2.0,
                                  fieldShape='square',signalDots='same',
                                  noiseDots='position',dotLife=3,
                                  color=[-1.0,-0.7366,-0.7529], colorSpace='rgb',
                                  opacity=1, depth=-1.0)
            stim_object =[grating,dots]

        stim_object_ls.append(stim_object)


    # Creating backgroung (bg) and foreground (fg) colors  per epoch
    bg_ls = list()
    fg_ls = list()
    for e in range(stimdict["EPOCHS"]):

        # Setting stimulus backgroung (bg) and foreground (fg) colors
        try:
            if stimdict["lum_vector"]:                    #Pradeep
                bg = set_intensity(e,lum_gaussian[e])
                fg = set_intensity(e,lum_gaussian[e])
                bg_ls.append(bg)
                fg_ls.append(fg)

            elif stimdict["lum"][e] == 111:
                # Gamma correction and 6-bit depth transformation
                bg = set_intensity(e,stimdict["bg"][e])
                fg = set_intensity(e,stimdict["fg"][e])
                bg_ls.append(bg)
                fg_ls.append(fg)

            elif stimdict["lum"][e] or stimdict["contrast"][e]:
                # Gamma correction and 6-bit depth transformation
                bg = set_bgcol(stimdict["lum"][e],stimdict["contrast"][e])
                fg = set_fgcol(stimdict["lum"][e],stimdict["contrast"][e])
                bg_ls.append(bg)
                fg_ls.append(fg)

            else:
                # Gamma correction and 6-bit depth transformation
                bg = set_intensity(e,stimdict["bg"][e])
                fg = set_intensity(e,stimdict["fg"][e])
                bg_ls.append(bg)
                fg_ls.append(fg)

        except:
            # Gamma correction and 6-bit depth transformation
            bg = set_intensity(e,stimdict["bg"][e])
            fg = set_intensity(e,stimdict["fg"][e])
            bg_ls.append(bg)
            fg_ls.append(fg)

##############################################################################
############################ NIDAQ CONFIGURATION #############################
##############################################################################

    # Initialize Time
    global_clock = core.Clock()

    # Timer initiation
    duration_clock = global_clock.getTime() # it will be reset at every epoch


    if dlp.OK:
        print('DLP used')

        counterTaskHandle = daq.TaskHandle(0)
        pulseTaskHandle = daq.TaskHandle(0)
        counterChannel = config.COUNTER_CHANNEL
        pulseChannel = config.PULSE_CHANNEL
        maxRate = config.MAXRATE

        # data from NIDAQ counter
        data = daq.uInt32(1)
        lastDataFrame = -1
        lastDataFrameStartTime = 0

        #DAQ SETUP FOR IMAGING SYNCHRONIZATION
        try:
            # DAQmx Configure Code
            daq.DAQmxCreateTask("2",daq.byref(counterTaskHandle))
            daq.DAQmxCreateCICountEdgesChan(counterTaskHandle,counterChannel,
                                            "",daq.DAQmx_Val_Rising,0,
                                            daq.DAQmx_Val_CountUp)
            daq.DAQmxCreateTask("1",daq.byref(pulseTaskHandle))
            daq.DAQmxCreateCOPulseChanTime(pulseTaskHandle,pulseChannel,
                                           "",daq.DAQmx_Val_Seconds,
                                           daq.DAQmx_Val_Low,0,0.05,0.05)

            # DAQmx Start Code
            daq.DAQmxStartTask(counterTaskHandle) # Reading any coming frame.
            daq.DAQmxStartTask(pulseTaskHandle)   # Sending trigger to mic.

            # Reads incoming signal from microscope computer and stores it to
            # 'data'. A rising edge is send every new frame the microscope
            # starts to record, thus the 'data' variable is incremented
            daq.DAQmxReadCounterScalarU32(counterTaskHandle,1.0,
                                          daq.byref(data), None)

            # Do we need that here? Check it with hardware.
            # Checks if new frame is being imaged.
            if (lastDataFrame != data.value):
                lastDataFrame = data.value
                lastDataFrameStartTime = global_clock.getTime()

        except daq.DAQError as err:
            print ("DAQmx Error: %s"%err)

    else:
        # When not using dlp (Checking the stimulus in th PCs monitor),
        # some varibales need to be defined anyways, although they are
        # not being change every frame.
        counterTaskHandle = None
        data = daq.uInt32(1)
        lastDataFrame = 0
        lastDataFrameStartTime = 0
        print('No DLP used')

##############################################################################
######### MAIN Loop which calls the functions to draw stim on screen #########
##############################################################################

    # Pause between sending the trigger to microscope and displaying stimuli
    # For not presenting the simuli during aninitial increase in fluorescence
    # that happens sometimes when the microscope starts scanning
    print('Microscope scanning started')
    print('5s pause...')
    time.sleep(5)
    print('Stimulus started')
    print('##############################################')

    # Main Loop: dit diplays the stimulus unless:
        # keyboard key is pressed (manual stop)
        # stop condition becomse "True"
    while not (len(event.getKeys()) > 0 or stop):
        #print(f'WHILE LOOP STARTS: {global_clock.getTime()}')

        # choose next epoch
        try:
            (epoch,current_index) = choose_epoch(shuffle_index,stimdict["RANDOMIZATION_MODE"],
                                             stimdict["EPOCHS"],current_index)
        except:
            (epoch,current_index) = choose_epoch(shuffle_index,stimdict['randomize'][0],
                                             stimdict["EPOCHS"],current_index) # Seb, temp for old stimulus design

        # Data for Output file
        out.boutInd = out.boutInd + 1
        out.epochchoose = epoch

        # Reset epoch timer
        duration_clock = global_clock.getTime()
        print(f'STIM SELECTION STARTS: {global_clock.getTime()}')
        try:


            # Functions that draw the different stimuli
            if stimdict["stimtype"][0]== "GaussCircle":    #Pradeep: Please keep this at top because there is only one epoch in my stimuli file as it runs for 1000s of luminace values and it does not make sense to write the parameters for each epoch

                (out, lastDataFrame, lastDataFrameStartTime) = stimuli.gaussian_flash(exp_Info,bg_ls,fg_ls,stim_texture_ls[epoch],noise_array_ls[epoch],stimdict,epoch, win, global_clock,duration_clock,outFile,
                                                                out,stim_object_ls[0],dlp.OK, viewpos, data, counterTaskHandle, lastDataFrame, lastDataFrameStartTime)

            elif stimdict["stimtype"][epoch] == "SSR":

                (out, lastDataFrame, lastDataFrameStartTime) = stimuli.standing_stripes_random(exp_Info,bg_ls,fg_ls,stimdict,epoch, win, global_clock,duration_clock,outFile,
                                                                out,stim_object_ls[epoch],dlp.OK,counterTaskHandle,data, lastDataFrame, lastDataFrameStartTime)

            elif stimdict["stimtype"][epoch][-1]== "C":

                (out, lastDataFrame, lastDataFrameStartTime) = stimuli.field_flash(exp_Info,bg_ls,fg_ls,stim_texture_ls[epoch],noise_array_ls[epoch],stimdict,epoch, win, global_clock,duration_clock,outFile,
                                                                out,stim_object_ls[epoch],dlp.OK, viewpos, data, counterTaskHandle, lastDataFrame, lastDataFrameStartTime)

            elif stimdict["stimtype"][epoch][-1]== "NC":

                (out, lastDataFrame, lastDataFrameStartTime) = stimuli.field_flash(exp_Info,bg_ls,fg_ls,stim_texture_ls[epoch],noise_array_ls[epoch],stimdict,epoch, win, global_clock,duration_clock,outFile,
                                                                out,stim_object_ls[epoch],dlp.OK, viewpos, data, counterTaskHandle, lastDataFrame, lastDataFrameStartTime)

            elif stimdict["stimtype"][epoch][-1]== "R":

                (out, lastDataFrame, lastDataFrameStartTime) = stimuli.field_flash(exp_Info,bg_ls,fg_ls,stim_texture_ls[epoch],noise_array_ls[epoch],stimdict,epoch, win, global_clock,duration_clock,outFile,
                                                                out,stim_object_ls[epoch],dlp.OK, viewpos, data, counterTaskHandle, lastDataFrame, lastDataFrameStartTime)
            
            elif stimdict["stimtype"][epoch] == "DS":
                #print(f'FUNCTION CALLED: {global_clock.getTime()}')
                (out, lastDataFrame, lastDataFrameStartTime) = stimuli.drifting_stripe(exp_Info,bg_ls,fg_ls,stimdict,epoch, win, global_clock,duration_clock,outFile,
                                                                out,stim_object_ls[epoch],dlp.OK, viewpos, data, counterTaskHandle, lastDataFrame, lastDataFrameStartTime)

                
            elif stimdict["stimtype"][epoch] == "ADS":
                #print(f'FUNCTION CALLED: {global_clock.getTime()}')
                (out, lastDataFrame, lastDataFrameStartTime) = stimuli.drifting_stripe_arbitrary_dir(bg_ls,fg_ls,stimdict,epoch, win, global_clock,duration_clock,outFile,
                                                                out,stim_object_ls[epoch],dlp.OK, viewpos, data, counterTaskHandle, lastDataFrame, lastDataFrameStartTime)


            elif stimdict["stimtype"][epoch] == "N":

                print(f"printing epoch in noise stim: {epoch}")

                (out, lastDataFrame, lastDataFrameStartTime) = stimuli.stim_noise(exp_Info,bg_ls,stim_texture,stimdict,epoch, win, global_clock,duration_clock,outFile,
                                                                out,stim_object_ls[epoch],dlp.OK,counterTaskHandle,data, lastDataFrame, lastDataFrameStartTime)
                #(out, lastDataFrame, lastDataFrameStartTime) = stimuli.stim_noise(bg_ls,stim_texture_ls[epoch-1],stimdict,epoch, win, global_clock,duration_clock,outFile,
                #out,stim_object_ls[epoch],dlp.OK,counterTaskHandle,data, lastDataFrame, lastDataFrameStartTime)
            elif stimdict["stimtype"][epoch] == "shift_noise":

                (out, lastDataFrame, lastDataFrameStartTime) = stimuli.h_res_noise(exp_Info,bg_ls,stim_texture_ls[epoch],stimdict,epoch, win, global_clock,duration_clock,outFile,
                                                                    out,stim_object_ls[epoch],dlp.OK,counterTaskHandle,data, lastDataFrame, lastDataFrameStartTime,dirs=copy_moves)
            elif stimdict["stimtype"][epoch] == "Noise_G":

                (out, lastDataFrame, lastDataFrameStartTime) = stimuli.sinusoid_grating_noise(exp_Info,30000,viewpos,stim_texture_ls[epoch-1],stimdict,epoch,win,global_clock,duration_clock,outFile
                                                                                      ,out,stim_object_ls[epoch],dlp.OK,counterTaskHandle,data, lastDataFrame, lastDataFrameStartTime)
            elif stimdict["stimtype"][epoch][-1:] == "G":

                (out, lastDataFrame, lastDataFrameStartTime)= stimuli.noisy_grating(exp_Info,_useNoise,_useTex,viewpos,bg_ls,stim_texture_ls[epoch],noise_array_ls[epoch],stimdict,epoch, win, global_clock,duration_clock,outFile,
                                                                out,stim_object_ls[epoch],dlp.OK,counterTaskHandle,data, lastDataFrame, lastDataFrameStartTime)

            elif stimdict["stimtype"][epoch] == "DG":

                (out, lastDataFrame, lastDataFrameStartTime)= stimuli.dotty_grating(exp_Info,_useNoise,_useTex,viewpos,bg_ls,stim_texture_ls[epoch],stimdict,epoch, win, global_clock,duration_clock,outFile,
                                                                out,stim_object_ls[epoch][0],stim_object_ls[epoch][1],dlp.OK,counterTaskHandle,data, lastDataFrame, lastDataFrameStartTime)


            else: 
                print(stimdict["stimtype"][epoch])
                raise StimulusError(stimdict["stimtype"][epoch],epoch)

            # Irregular stop conditions:
            # "and not stimdict["MAXRUNTIME"]==0" is an quick fix to test stim
            # on dlp without mic. Important for SEARCH Stimulus
            if (dlp.OK and (global_clock.getTime() - lastDataFrameStartTime > 1)
                and not stimdict["MAXRUNTIME"]==0):
                raise MicroscopeException(lastDataFrame,lastDataFrameStartTime,global_clock.getTime())
            elif (dlp.OK and (global_clock.getTime() >= stimdict["MAXRUNTIME"])
                  and not stimdict["MAXRUNTIME"]==0):
                raise StimulusTimeExceededException(stimdict["MAXRUNTIME"],global_clock.getTime())
            elif (global_clock.getTime() >= MAXRUNTIME) and not stimdict["MAXRUNTIME"]==0:
                raise GlobalTimeExceededException(MAXRUNTIME,global_clock.getTime())

        # Real Errors
        except StimulusError as e:
            print ('Stimulus function could not be executed. Stimtype:', e.type)
            print ('At epoch:', e.epoch)
            raise
        except daq.DAQError as err:
            print ("DAQmx Error: %s"%err)
        # Irregular stop conditions:
        except MicroscopeException or StimulusTimeExceededException or GlobalTimeExceededException as e:
            pass
            print ("A stop condition became true: " )
            print ("Time of %s was exceeded by current time %s at microscope frame %s. Maybe better use testmode (no DLP)?" %(e.spec_time,e.time,e.frame))
            print (e)
            stop = True
        # Manual stop from stimulus:
        except StopExperiment:
            print('##############################################')
            print ("Stopped experiment manually")
             # fake key-press to stop experiments through event listener
            event._onPygletKey(key.END,key.MOD_CTRL)

##############################################################################
    # Save data
    outFile.close()
    #save_main_setup(config.OUT_DIR) #OLD, deprecated
    # out.save_outfile(config.OUT_DIR)#OLD, deprecated

    # DAQmx Stop Code
    if counterTaskHandle:
        clearTask(counterTaskHandle)
    if counterTaskHandle:
        clearTask(pulseTaskHandle)

    

    #Saving movie frames
    if exp_Info['saving_movie_frames']: #Not recomended for usual recordings but just for examples of short duration
        #Saving movie frames
        print('Saving movie frames...')
        folder_path = config.OUT_DIR
        file_name =f'stimulus.gif'
        saving_path =os.path.join(folder_path,file_name)
        try:
            win.saveMovieFrames(saving_path)
        except:
            print('>>> win.saveMovieFrames failed to generate the .gif file \n check the epoch specifig movie folder')
                    # Set the input folder containing .png files
        input_folder = os.path.join(config.OUT_DIR,'Last_stim_movie_frames')
        for folder_path, _, _ in os.walk(input_folder):
            # Set the saving path for the movie file (e.g., .mp4)
            saving_path = os.path.join(folder_path, 'stimulus.mp4')
            # Set the frames per second (fps) for the movie
            fps = config.FRAMERATE
            # Create the movie from .png files
            create_movie_from_png(folder_path, saving_path, fps)

        win.close()
        core.quit()
    
    else:

        # Stop
        print ("Write out ... closed!")

        win.close()
        core.quit()


def clearTask(taskHandle):
    """
    Clears a task from the card.
    """
    daq.DAQmxStopTask(taskHandle)
    daq.DAQmxClearTask(taskHandle)

if __name__ == "__main__":
    main()

# %%
