#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
from modules import main 
from modules import config
from psychopy import gui

def user(user_ID):
    #Coded being executed from the terminal
    print('##############################################')
    print(f'Running pyVisualStim. Good luck {user_ID}!')

    #Creating folder
    user_folder = config.OUT_DIR
    if not os.path.exists(user_folder):
        os.mkdir(user_folder)
        #Setting user information for the first time
        mainfile_name_temp = os.path.join(user_folder,'current_recording.txt')
        mainfile_temp = open(mainfile_name_temp, 'w')
        mainfile_temp.write(f'ID,value\n')
        mainfile_temp.write(f'USER_ID,{user_ID}\n')
        mainfile_temp.write(f'SUBJECT_ID,\n')
        mainfile_temp.write(f'EXP_NAME,\n')

    #Asking for stimulus
    file_path = gui.fileOpenDlg('./stimuli_collection')
    main(file_path[0])


if __name__ == "__main__":
    globals()[sys.argv[1]](sys.argv[2]) # Makes possible to run the user() function and input the unser_name variable in the command line
    user()

    # #For running from the terminal
    # "In the terminal write: python run.py user <choose_user_name>"
    # #user("seb"), for example. 

    

    
