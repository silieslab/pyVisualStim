#!/usr/bin/env python
# -*- coding: utf-8 -*-

#############################################################################
# This script contains all test functions for some helper functions
# Eventu<lly, every helper function has to have its test function
# DIsclaimer, not all test functions have "assert" statements. The funtionality
# for some of them dependo on what is succesfully printed on the screen as an
# image.

def test_window_3masks():
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

    import psychopy
    from psychopy import visual,core,logging,event, gui, monitors
    from psychopy.visual.windowwarp import Warper # perspective correction
    from helper import window_3masks

    # Windows creatinon
    _width = 400
    _height = 400
    _monitor = 'testMonitor'

    win = visual.Window(fullscr = False, monitor=_monitor,
                                size = [_width,_height], viewScale = [1,1],
                                screen = 0,
                                color=[-1,-1,-1],useFBO = True,allowGUI=False,
                                viewOri = 0.0)


    win_mask_ls  = window_3masks(win=win,_monitor = 'testMonitor')

    # Stimulus creation #
    circle_stim = visual.Circle(win=win, radius=400, units='pix',
                        fillColor=[1,1,1], edges = 128)


    # Stimulus drawing
    circle_stim.draw()
    win.flip()
    win_mask_ls[0].flip()
    win_mask_ls[1].flip()
    win_mask_ls[2].flip()

    core.wait(3)

    win.close()
    win_mask_ls[0].close()
    win_mask_ls[1].close()
    win_mask_ls[2].close()
    '''
    # Printing useful info
    print(f"The used monitor '{win.monitor.name}' has a resolution of: {win.monitor.getSizePix()} pixels")
    print(f"Main screnn located at: {win.pos} pixels")
    print(type(win.size[0]))
    print(win.size[1])
    '''




def test_window_4masks(_width=400,_height= 400,_xpos=568,_ypos=232):

    '''
    NOT FINISHED, check TODO

    Draws mask of for a screen, so that the smaller screen is insede the big
    one like the scheme: the center (C) at the top of the small screen is in
    the center of the big one and that the big screen fills all the resolution
    in the x-axis.


                        #######################
                        #                     #
                        #                     #
                        #     #####C#####     #
                        #     #         #     #
                        #     #         #     #
                        #     #         #     #
                        #     ###########     #
                        #                     #
                        #######################


    Screen size of my laptop in psychopy = 1536,864
    '''


    import psychopy
    from psychopy import visual,core,logging,event, gui, monitors
    from psychopy.visual.windowwarp import Warper # perspective correction
    from helper import max_horizontal_angle




    ########################## Windows creation ############################

    win = visual.Window(fullscr = False, monitor='testMonitor',
                                size = [_width,_height], viewScale = [1,1],
                                screen = 0, pos = [_xpos,_ypos],
                                color=[-1,-1,-1],useFBO = True,allowGUI=False,
                                viewOri = 0.0)

    scr_width = win.scrWidthCM
    scr_distance = win.scrDistCM
    maxhorang = max_horizontal_angle(scr_width, scr_distance)

    # TO DO: midified the aboved code to move the small screen up 15 deg and add a 4th mask
    '''
    win_mask_sizes =[[_width,_height/2],[_width/4,_height],[_width/4,_height]]
    win_mask_positions =[[_xpos,_ypos],[_xpos,_ypos],[_xpos+(3*(_width/4)),_ypos]]

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


    ########################## Stimulus creation ############################

    circle_stim = visual.Circle(win=win, radius=400, units='pix',fillColor=[1,1,1], edges = 128)


     ########################## Stimulus drawing ############################
    circle_stim.draw()
    win.flip()
    win_mask_0.flip()
    win_mask_1.flip()
    win_mask_2.flip()

    core.wait(5)

    win.close()
    win_mask_0.close()
    win_mask_1.close()
    win_mask_2.close()


    ########################### Printing useful info ########################
    print(f"The used monitor '{win.monitor.name}' has a resolution of: {win.monitor.getSizePix()} pixels")
    print(f"Main screnn located at: {win.pos} pixels")
    '''
