import cv2
import numpy as np
import keyboard
import pyautogui as pg
from PIL import ImageGrab
import time

i = 0
lable = "Dataset\dataframe.csv"
with open(lable, 'w') as file:
    file.write("image_path,w,a,s,d\n")
    while True:

        # srint screen
        screenshot = cv2.cvtColor(np.array(ImageGrab.grab(bbox=(450,130,1470,875)), dtype='uint8'), cv2.COLOR_BGR2GRAY)
        screenshot = cv2.resize(screenshot, (640, 480))
        # show prints
        cv2.imshow('Video', screenshot)

        # save the images
        i += 1
        path = "Dataset\Data\image_" + str(i) + ".png"
        cv2.imwrite(path, screenshot)

        # read keyboard
        input = ''

        if(keyboard.is_pressed('a') and keyboard.is_pressed('w')):
            input = '1,1,0,0'
        elif(keyboard.is_pressed('w') and keyboard.is_pressed('d')):
            input = '1,0,0,1'
        elif(keyboard.is_pressed('s') and keyboard.is_pressed('d')):
            input = '0,0,1,1'
        elif(keyboard.is_pressed('a') and keyboard.is_pressed('s')):
            input = '0,1,1,0'
        elif(keyboard.is_pressed('a')):
            input = '0,1,0,0'
        elif(keyboard.is_pressed('s')):
            input = '0,0,1,0'
        elif(keyboard.is_pressed('d')):
            input = '0,0,0,1'
        elif(keyboard.is_pressed('w')):
            input = '1,0,0,0'
        else:
            input = '0,0,0,0'

        print(input)
        image_path = "/image_" + str(i) + ".png"
        file.write(image_path + "," + str(input) + "\n")

        if(keyboard.is_pressed('p')):
            time.sleep(20)
        elif(keyboard.is_pressed('backspace')):
            break
        
        cv2.waitKey(200)

    cv2.destroyAllWindows()


