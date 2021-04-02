import numpy as np
from PIL import Image

from handler_test import ModelHandler

image = np.array(Image.open("human_pose.jpg").convert('RGB'))


handler = ModelHandler()

handler.handle(image, "context")