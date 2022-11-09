import cv2
import numpy as np
import keyboard
import pyautogui as pg

i = 0
keys = []
while True:

    # srint screen
    screenshot = np.array(pg.screenshot())

    # show prints
    cv2.imshow('Video', screenshot)

    # save the images
    i += 1
    path = "Dataset\Data\image_" + str(i) + r".png"
    cv2.imwrite(path, screenshot)

    # read keyboard
    input = ''
    
    if(keyboard.is_pressed('a') and keyboard.is_pressed('w')):
        input = 'aw'
    elif(keyboard.is_pressed('w') and keyboard.is_pressed('d')):
        input = 'wd'
    elif(keyboard.is_pressed('s') and keyboard.is_pressed('d')):
        input = 'sd'
    elif(keyboard.is_pressed('a') and keyboard.is_pressed('s')):
        input = 'as'
    elif(keyboard.is_pressed('a')):
        input = 'a'
    elif(keyboard.is_pressed('s')):
        input = 's'
    elif(keyboard.is_pressed('d')):
        input = 'd'
    elif(keyboard.is_pressed('w')):
        input = 'w'
    else:
        input = 'nk'

    keys.append(input)
    print(input)

    # escape condition
    if cv2.waitKey(200) & 0xFF == 112: 
        break

print(keys)
cv2.destroyAllWindows()


