# import torch
# import numpy as np
# import cv2

# from PIL import Image

# from model import PoseEstimationWithMobileNet
# from val import normalize, pad_width
# from load_state import load_state

# height_size, scale, stride = 256, None, 8
# model = PoseEstimationWithMobileNet()
# checkpoint = torch.load("checkpoint_iter_370000.pth", map_location='cpu')
# load_state(model, checkpoint)

# img = np.array(Image.open("human_pose.jpg").convert('RGB'))
# pad_value=(0, 0, 0)
# img_mean=np.array([128, 128, 128], np.float32)
# img_scale=np.float32(1/256)
# # img = Image.open(io.BytesIO(data[0]["body"])).convert('RGB')
# # nparr = np.frombuffer(data[0]["body"], np.uint8)
# # img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # cv2.IMREAD_COLOR in OpenCV 3.1
# # print("img", img)
# height, width, _ = img.shape
# # print(type(height_size), type(height))
# scale = height_size / height
# scale = scale

# scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
# scaled_img = normalize(scaled_img, img_mean, img_scale)
# min_dims = [height_size, max(scaled_img.shape[1], height_size)]
# padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)
# pad = pad

# tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
# with torch.no_grad():
#     print(model(tensor_img))
#     traced_cell = torch.jit.trace(model, (tensor_img))
# torch.jit.save(traced_cell, "PoseEstimation_model.pth")



import torch
import io

from model import PoseEstimationWithMobileNet
from load_state import load_state


model = PoseEstimationWithMobileNet()
checkpoint = torch.load("checkpoint_iter_370000.pth", map_location='cpu')
load_state(model, checkpoint)

m = torch.jit.script(model)

# Save to file
torch.jit.save(m, 'PoseEstimation_model.pt')

