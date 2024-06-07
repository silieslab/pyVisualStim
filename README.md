# pyVisualStim

Code to generate diverse visual stimulation on a screen.

### SOFTWARE INSTALLATION
To run the code, you need additional packages

Step-by-step package installation.

1. Install anaconda and craete an environment (e.g 2pVStim)
- >> conda create --name <ENV_NAME> python = 3.9

2. Go inside the environment (activate it)
- >> conda activate <ENV_NAME>
- check the default (already installed) packages with: >> conda list

2. Install Psychopy3
- pip install psychopy (depending on you OP, other pacakges will need to be install first or use another package platforms like conda-forge. Check errors)
- pip install pyDAQmx (check the versin that is compatible for your OP)
- pip instll h5py
- check the  installed packages with: >> conda list


In order to use it together with other devices and project the sitmulation using a Texas Instrument projector, move to HARWARE INSTALLATION section.

### HARDWARE INSTALLATION

1. NI-DAQmx installation (drivers for the DAC)
2. DLP gui installation (DLPLCR4500GUI)


### RUNNING pyVisualStim

1. Create an output folder somewhere in your PC with a subfolder for an USER (e.g. "C:\Documents\temp_pyVisualStim_OutputFiles\USER_NAME")
Here the folder <Temp_pyVisualStim_OutputFiles> contains the user <USER_NAME>
For storage and visualization of movie frames (video), is recommended to also have a subsubfolfer named <Last_stim_movie_frames> in the user subfolder
2. In this folder, insert the default "current_recording" file. (see included txt in the files of this code)
The variable names there can be modified as wished per every recording
3. Run the stimulation using the terminal
- Activate the proper environment: >> conda activate <ENV_NAME>
- Go to the main code folder (pyVisualStim): cd <CODE_PATH>
- Type to run the code: >> python run.py user <USER_NAME>
- You will be ask to: first, choose the "current_recording", and second, choose the stimulus txt file to run (navigate to those files)

### ASPECTS TO CONSIDER BEFORE RUNNING THE CODE
1. Set the refresh rate of yor monitor
- In the config.py file, set the refresh rate of teh monitor / projector being used.

2. PsychoPy Monitor Center 
- add your monitor and set the screen size, distacnde to screen and resolution