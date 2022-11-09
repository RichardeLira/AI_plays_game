import os 

BASE_PATH   = r"C:\Users\richa\OneDrive\Documentos\Computer Vision\AI_plays_game\DataSet"
IMAGES_PATH = os.path.sep.join(BASE_PATH, "Data")
LABEL_PATH  = os.path.sep.join(BASE_PATH, "Labels") 

BASE_OUTPUT = r"C:\Users\richa\OneDrive\Documentos\Computer Vision\AI_plays_game\Output"

MODEL_OUTPUT = os.path.sep.join(BASE_OUTPUT, "IA_Car.pth")

BATH_SIZE  = 32
LEARN_RATE = 1e-3 
NUM_EPOCH  = 20 