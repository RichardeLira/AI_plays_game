import cv2
import time
import pytorch
import keyboard
import numpy as np
import pydirectinput
from PIL import ImageGrab


# device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# load model
model = torch.load(Config.MODEL_PATH).to(Config.DEVICE)
model.eval()

# define normalization transforms
transforms = transforms.Compose(
    [transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean= Config.MEAN, std=Config.STD)]
)

time.sleep(5)
label = [0,0,0,0]
while True:

    # print screen
    screenshot = cv2.cvtColor(np.array(ImageGrab.grab(bbox=(450,160,1470,875)), dtype='uint8'), cv2.COLOR_BGR2RGB)

    # show prints
    cv2.imshow('Video', screenshot)

    # resize image and transpose layot for pytorch
    screenshot = cv2.resize(screenshot, (299, 299))
    screenshot = screenshot.transpose((2, 0, 1))

    # convert image to tensor and normalize it
    screenshot = torch.from_numpy(screenshot)
	screenshot = transforms(screenshot).to(Config.DEVICE)
	screenshot = screenshot.unsqueeze(0)

    # predict the labels
    label = model(screenshot)
    label2 = torch.nn.Softmax(dim=-1)(label)

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