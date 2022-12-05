import cv2
import numpy as np
import keyboard
import pyautogui as pg
from PIL import ImageGrab
import time
import pandas as pd
import pickle as p

dt = pd.DataFrame()
i = 0

time.sleep(10)
while True:

    # print screen
    screenshot = cv2.cvtColor(np.array(ImageGrab.grab(bbox=(450,160,1470,875)), dtype='uint8'), cv2.COLOR_BGR2RGB)
    screenshot = cv2.resize(screenshot, (299, 299))

    # show prints
    cv2.imshow('Video', screenshot)

    cv2.imwrite("Dataset\Data\image_" + str(i) + ".png", screenshot)
    i += 1

    # read keyboard
    input = []

    if(keyboard.is_pressed('a') and keyboard.is_pressed('w')):
        input = [1,1,0,0]
    elif(keyboard.is_pressed('w') and keyboard.is_pressed('d')):
        input = [1,0,0,1]
    elif(keyboard.is_pressed('s') and keyboard.is_pressed('d')):
        input = [0,0,1,1]
    elif(keyboard.is_pressed('a') and keyboard.is_pressed('s')):
        input = [0,1,1,0]
    elif(keyboard.is_pressed('a')):
        input = [0,1,0,0]
    elif(keyboard.is_pressed('s')):
        input = [0,0,1,0]
    elif(keyboard.is_pressed('d')):
        input = [0,0,0,1]
    elif(keyboard.is_pressed('w')):
        input = [1,0,0,0]
    else:
        input = [0,0,0,0]

    # add row in dataframe
    dt = dt.append({'image':screenshot, 'lable':input}, ignore_index=True)
    print(input)

    # key binds
    if(keyboard.is_pressed('p')):
        time.sleep(20)
    elif(keyboard.is_pressed('backspace')):
        break
        
    cv2.waitKey(200)

# save dataset
with open('Dataset/dataset.pkl', 'wb') as f:
    p.dump(dt, f)

cv2.destroyAllWindows()