import cv2
import time
import keyboard
import numpy as np
import pydirectinput
from PIL import ImageGrab

time.sleep(5)
label = [0,0,0,0]
while True:

    # print screen
    screenshot = cv2.cvtColor(np.array(ImageGrab.grab(bbox=(450,160,1470,875)), dtype='uint8'), cv2.COLOR_BGR2RGB)
    screenshot = cv2.resize(screenshot, (299, 299))

    # show prints
    cv2.imshow('Video', screenshot)

    # key press
    if(label[0] == 1):
        pydirectinput.keyDown('w')
    else:
        pydirectinput.keyUp('w')
    if(label[1] == 1):
        pydirectinput.keyDown('a')
    else:
        pydirectinput.keyUp('a')
    if(label[2] == 1):
        pydirectinput.keyDown('s')
    else:
        pydirectinput.keyUp('s')
    if(label[3] == 1):
        pydirectinput.keyDown('d')
    else:
        pydirectinput.keyUp('d')

    # key binds
    if(keyboard.is_pressed('p')):
        time.sleep(20)
    elif(keyboard.is_pressed('backspace')):
        break

    cv2.waitKey(1)

cv2.destroyAllWindows()