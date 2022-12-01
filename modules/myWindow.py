from psychopy import visual

class myWindow(visual.Window):
    
    def onResize(width, height):
        """A default resize event handler.
        This default handler updates the GL viewport to cover the entire
        window and sets the ``GL_PROJECTION`` matrix to be orthogonal in
        window space.  The bottom-left corner is (0, 0) and the top-right
        corner is the width and height of the :class:`~psychopy.visual.Window`
        in pixels.
        Override this event handler with your own to create another
        projection, for example in perspective.
        """
        if height == 0:
            height = 1
        GL.glViewport(0, 0, width, height)
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glLoadIdentity()
        # GL.glOrtho(-1, 1, -1, 1, -1, 1)
        
        GL.gluPerspective(90, 1.0 * width / height, 0.1, 100.0)
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glLoadIdentity()