import cv2
import time
import torch
import keyboard
import numpy as np
import pydirectinput
from torchvision import transforms
from PIL import ImageGrab
 

import os 
 
#def predict():
from torch.nn import Module
from torch.nn import Sequential
from torch.nn import Identity
from torch.nn import Dropout
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU



class Self_Car_Driving(Module): 
    
    def __init__(self,base_model,num_classes):
      super(Self_Car_Driving,self).__init__()


      self.base_model  = base_model
      self.num_classes = num_classes 
        

      self.classifier = Sequential (
          
          Linear(base_model.fc.in_features, 512),
			    ReLU(),
			    Dropout(),
			    Linear(512, 512),
			    ReLU(),
			    Dropout(),
			    Linear(512, self.num_classes)
      ) 

      self.base_model.fc = Identity()

    
    def forward(self,X):
      features = self.base_model(X)
      classLogits = self.classifier(features)
      return classLogits




# define the base path to the input dataset and then use it to derive
# the path to the input images and annotation CSV files
BASE_PATH = r"C:\Users\richa\OneDrive\Documentos\Computer Vision\AI_plays_game\AI_plays_game"
MODEL_PATH = os.path.sep.join([BASE_PATH, "model_TEST_64.pth"])
# device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# MODEL_PATH = r"C:\Users\richa\OneDrive\Documentos\Computer Vision\AI_plays_game\AI_plays_game\model_TEST.pth"
# load model
model = torch.load(MODEL_PATH).to(DEVICE)
model.eval()
# define normalization transforms
transforms = transforms.Compose(
    [transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean= [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)

import time 

# time.sleep(5)
label = [0,0,0,0]

frame = 0
while True:
    frame +=1
    print(frame)    
    # print screen
    screenshot = cv2.cvtColor(np.array(ImageGrab.grab(bbox=(450,120,1470,850)), dtype='uint8'), cv2.COLOR_BGR2RGB)
    # # show prints
    # cv2.imshow('Video', screenshot)
    # resize image and transpose layot for pytorch
    screenshot = cv2.resize(screenshot, (299, 299))
    screenshot = screenshot.transpose((2, 0, 1))
    # convert image to tensor and normalize it
    screenshot = torch.from_numpy(screenshot)
    screenshot = transforms(screenshot).to(DEVICE)
    screenshot = screenshot.unsqueeze(0)
    # predict the labels
    label = model(screenshot)
    y_pred_softmax = torch.log_softmax(label, dim =1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
    
 
        
    if(y_pred_tags == 4):
        pydirectinput.keyDown('w')
        pydirectinput.keyDown('a')
    else:
        pydirectinput.keyUp('w')
        pydirectinput.keyUp('a')
    if(y_pred_tags == 5):
        pydirectinput.keyDown('w')
        pydirectinput.keyDown('d')
    else:
        pydirectinput.keyUp('w')
        pydirectinput.keyUp('d')
    
    if(y_pred_tags == 6):
        pydirectinput.keyDown('s')
        pydirectinput.keyDown('a')
    else:
        pydirectinput.keyUp('s')
        pydirectinput.keyUp('a')
    
    if(y_pred_tags == 7):
        pydirectinput.keyUp('w')
        pydirectinput.keyUp('a')
        pydirectinput.keyUp('s')
        pydirectinput.keyUp('d')
    if(y_pred_tags == 0):
        pydirectinput.keyDown('w')
    else:
        pydirectinput.keyUp('w')
    if(y_pred_tags == 1):
        pydirectinput.keyDown('a')
    else:
        pydirectinput.keyUp('a')
    if(y_pred_tags == 2):
        pydirectinput.keyDown('s')
    else:
        pydirectinput.keyUp('s')
    if(y_pred_tags == 3):
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


#if __name__ == '__main__':
#    predict()