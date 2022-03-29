from tkinter import *
import numpy

class GUI(Canvas, object):
    def __init__(self,root,a=22, b=22, unit=40, size=15):
        Canvas.__init__(root,bg="black",height=a*unit,width=b*unit)
        self.allObject = list()
