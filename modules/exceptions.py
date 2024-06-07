#!/usr/bin/env python
# -*- coding: utf-8 -*-


    
class StimulusError(Exception):
    """ Thrown, if there is an issue with the chosen Stimulustype 
    
        :param type: The Stimtype
        :type type: int
        :param epoch: The epoch, in which the error occured
        :type epoch: int
    
    """
    def __init__(self,type,epoch):
        self.type = type
        self.epoch = epoch
    def __str__(self):
        return repr(self.type,self.epoch)
    
class StopExperiment(Exception):
    """ Used to halt experiment quickly"""
    pass

  
class MicroscopeException(Exception):
    """
        Either a microscope synch. signal was not send, not recognized or lost.
    """
    def __init__(self,lastdataframe,spec_time,time):
        self.spec_time = spec_time
        self.time = time
        self.frame = lastdataframe
    def __str__(self):
        return repr(self)
         
class StimulusTimeExceededException(Exception):
    """
        The MAXRUNTIME specified in the stimulus file was exceeded.
    """
    def __init__(self,spec_time,time):
        self.spec_time = spec_time
        self.time = time
    def __str__(self):
        return repr(self)
        
                 
class GlobalTimeExceededException(Exception):
    """   
        The global MAXRUNTIME specified in ``config.py`` was exceeded.
    """
    def __init__(self,spec_time,time):
        self.spec_time = spec_time
        self.time = time
    def __str__(self):
        return repr(self)
        
