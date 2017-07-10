import numpy as np
from PIL import ImageGrab
import cv2

def GrabScreen(size=(0,50,800,650)):
    screen =  np.array(ImageGrab.grab(bbox=size))
    return cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)