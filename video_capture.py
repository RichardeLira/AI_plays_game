import cv2
import numpy as np
import keyboard
import pyautogui as pg

i = 0
while True:

    # srint screen
    screenshot = np.array(pg.screenshot())

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

    lable = "Dataset\Labels\image_" + str(i) + ".csv"
    with open(lable, 'w') as file:
        file.write(input)
    
    print(input)

    # escape condition
    if cv2.waitKey(200) & 0xFF == 112: 
        break

cv2.destroyAllWindows()


