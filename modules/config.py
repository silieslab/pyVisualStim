#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" This module contains global variables.

.. data:: MY_ID
    User identification
.. data:: OUT_DIR
    Directory for the stimulus output files

.. data:: MAINFILE_NAME
    Name for the metadata outputfile
.. data:: OUTFILE_NAME
    Name fo the data outputfile

.. data:: FRAMERATE
    The refresh rate of your monitor. Check PC monitor settings
.. data:: DISTANCE
    Distance from the viewer to the screen
.. data:: SCREEN_WIDTH
    Width of the projection are of the screen
.. data:: VIEWPOS_FILE
    Txt file if the size (x and y) and the position of the screen


.. data:: MAXRUNTIME
    By defaul 3600 seconds (60 mnin). Global time. Total duration of a recording.
    If this is exceded, stimulus presentation stops.
.. data:: SEED
    Seed number to be used in some pseudorandomization process in the main code

.. data:: COUNTER_CHANEL
    Where to read the counter of scanned frames from the microscope to the NI-DAQ
.. data:: PULSE_CHANNEL
    Where to send the trigger from the NI-DAQ to the microscope for start scanning


"""

import os
import pandas as pd
from datetime import datetime
from psychopy import gui

# For the current recording
#Hard coded path for every PC, must be put of any Github folder
print('>>> Select the file containing your recording IDs information <<<')
print('Stim OuputFiles will be saved in the same directory')
OUT_DIR = r'C:\Users\smolina\Documents' # Output files directory. Where to save them
IDs_FILE_DIR  = gui.fileOpenDlg(OUT_DIR)


#Reading current experimental info from file
try:
    OUT_DIR  = os.path.dirname(IDs_FILE_DIR[0])
    id_df = pd.read_csv(IDs_FILE_DIR[0], sep=',')
    ID_DICT ={}
    for key, value in zip(id_df['ID'],id_df['value']):
        ID_DICT[key] = value

except:
    print('>>> WARNING <<< ')
    print('You do not have a user folder and a current_recording.txt file. It is recomended to create one for future usage.')
    print('It should look like this example:\n')
    print('''ID,value\nUSER_ID,seb\nEXP_NAME,LC11_BL68362_2x_GCaMP6f\nSUBJECT_NUMBER,fly1\nTSERIES_NUMBER,001\nGENOTYPE,LC11_splitGal4_2x_GCaMP6f\nCONDITION,ExpLine\nSTIMULUS_ID,DQ100_30WB\nAGE,4\nSEX,f''')
    ID_DICT ={}
    ID_DICT['USER_ID'] = 'Input user name'
    ID_DICT['EXP_NAME'] = 'Input the experiment name' 
    ID_DICT['SUBJECT_ID'] = 'Input fly number'
    ID_DICT['TSERIES_NUMBER'] = 'Input Tseries number'
    ID_DICT['GENOTYPE'] = 'Input genotype name'
    ID_DICT['CONDITION'] = 'Input condition'
    ID_DICT['STIMULUS_ID'] = 'Input stimulus name'
    ID_DICT['AGE'] = 'Input age'
    ID_DICT['SEX'] = 'Input sex'
    print('>>> END OF WARNING <<< \n')
    

# Initializing USER_ID, SUBJECT_ID and EXP_NAME every new date and user
x = datetime.now()
DATE = x.strftime("%Y")+x.strftime("%m")+ x.strftime("%d")
TIME = x.strftime("%H")+x.strftime("%M")
ID_DICT['SUBJECT_ID'] = f"{DATE}-{ID_DICT['SUBJECT_ID']}"

# For output file configutation
SUBJECT_ID = ID_DICT['SUBJECT_ID']
TSERIES_NUMBER = ID_DICT['TSERIES_NUMBER']
METAFILE_NAME = f'{DATE}_{TIME}_{SUBJECT_ID}_{TSERIES_NUMBER}_meta_data' #Adds date and starting time to the files name
OUTFILE_NAME = f'{DATE}_{TIME}_{SUBJECT_ID}_{TSERIES_NUMBER}_stimulus_output'#Adds date and starting time to the files name

# For screen configuration
FRAMERATE = 60# Check refresh rate of your screen (here, PC monitor or projector)
DISTANCE = 5.36# Distance of the fly to the screen, For dlp 90deg: 5 cm, For ASUS monitor: 10, For dlp 45deg: ? (5.36?)
SCREEN_WIDTH = 9 # Width of the window's projection area, For dlp 90deg: 12, For ASUS monitor: 30,  For dlp 45deg: ? (9?)
VIEWPOS_FILE = 'viewpositions.txt' # Contains window' size and position on the screen
VIEWPOINT_X = 0.5
VIEWPOINT_Y = 0.5
WARP = 'spherical'
WIN_MASK = 0
MODE = 'patternMode' #'patternMode', 'videoMode'

# Other configurations
MAXRUNTIME = 3600
SEED = 54378  # To keep reproducibility among experiments >> DO NOT CHANGE this SEED number: (54378, original from 2020)

# For NIDAQ configuration
COUNTER_CHANNEL = "Dev2/ctr1" # or "Dev2/ctr1"
PULSE_CHANNEL = "Dev2/ctr0"  #or "Dev2/ctr0". Consider using not a counter but digital mode 'port1/line0' (digital channel)
MAXRATE = 10000.0 # Seb, currently unused

# For monitor color and luminance calibration:
CALIBRATE_GAMMA = 0 # 0 or 1
LUM_INPUTS =   [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0] # Psychopy units, range 0:1
LUM_MEASURED = [0, 0.05, 0.05,0.05,0.13,0.23,0.35,0.53,0.73,0.9,0.98] # Any units
LUM_MEASURED = [0.03, 0.05, 0.05,0.07,0.08,0.11,0.12,0.15,0.17,0.20,0.23] # Any units
if MODE == 'patternMode':
    GAMMA_LS = [1,1,1] # Gamma for each channel [R,G,B]
    COLOR_ON = [0,1,1] # 1 or 0 for [R,G,B]
elif MODE == 'videoMode':
    GAMMA_LS = [1,1,1.8] # Gamma for each channel [R,G,B]
    COLOR_ON = [0,0,1] # 1 or 0 for [R,G,B]

#For visualizing stimulus for presentations
SAVE_MOVIE = False
