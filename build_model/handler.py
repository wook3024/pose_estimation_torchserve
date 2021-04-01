import cv2
import numpy as np
import torch
import io
import json

from flask import jsonify
from PIL import Image
# from munch import  Munch
from ts.custom_handler.model import PoseEstimationWithMobileNet
from ts.custom_handler.keypoints import extract_keypoints, group_keypoints
from ts.custom_handler.load_state import load_state
from ts.custom_handler.pose import Pose, track_poses
from ts.custom_handler.val import normalize, pad_width

# custom handler file

# model_handler.py

"""
ModelHandler defines a custom model handler.
"""


from ts.torch_handler.base_handler import BaseHandler

class ModelHandler(BaseHandler):
    """
    A custom model handler implementation.
    """

    def __init__(self):
        self._context = None
        self.initialized = False
        self.explain = False
        self.target = 0
        self.pad = None
        self.scale = None
        self.height_size=256 
        self.video=0 
        self.images='' 
        self.cpu=True 
        self.track=0 
        self.smooth=1
        self.stride = 8
        self.upsample_ratio = 4
        self.previous_poses = None
        # self.model = PoseEstimationWithMobileNet()
        # checkpoint = torch.load("checkpoint_iter_370000.pth", map_location='cpu')
        # load_state(self.model, checkpoint)

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        self._context = context
        self.initialized = True
        #  load the model, refer 'custom handler class' above for details
        properties = context.system_properties
        # model_dir = properties.get("model_dir")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")
        # serialized_file = manifest['model']['serializedFile']
        # model_pt_path = os.path.join(model_dir, serialized_file)
        # if not os.path.isfile(model_dir):
        #     raise RuntimeError("Missing the model.pt file")


        # print("")    
        # print("model_pt_path", model_pt_path)
        # print("")

        # self.model = PoseEstimationWithMobileNet()
        self.model = torch.jit.load("/tmp/PoseEstimation/PoseEstimation_model.pt")
        # checkpoint = torch.load("/tmp/PoseEstimation/PoseEstimation_model.pth", map_location='cpu')
        # checkpoint = torch.load(model_dir)
        # load_state(self.model, checkpoint)
        

    def preprocess(self, data):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """
        # Take the input data and make it inference ready
        pad_value=(0, 0, 0)
        img_mean=np.array([128, 128, 128], np.float32)
        img_scale=np.float32(1/256)
        img = Image.open(io.BytesIO(data[0]["body"])).convert('RGB')
        nparr = np.frombuffer(data[0]["body"], np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # cv2.IMREAD_COLOR in OpenCV 3.1
        # print("img", img)
        height, width, _ = img.shape
        # print(type(self.height_size), type(height))
        scale = self.height_size / height
        self.scale = scale

        scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        scaled_img = normalize(scaled_img, img_mean, img_scale)
        min_dims = [self.height_size, max(scaled_img.shape[1], self.height_size)]
        padded_img, pad = pad_width(scaled_img, self.stride, pad_value, min_dims)
        self.pad = pad

        tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
        if not self.cpu:
            tensor_img = tensor_img.cuda()

        return tensor_img


    def inference(self, model_input):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """
        # Do some inference call to engine here and return output
        
        # model_output = self.model.forward(model_input)
        model_output = self.model(model_input)
        return model_output

    def postprocess(self, inference_output):
        """
        Return inference result.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        num_keypoints = 18
        kpt_names = ['nose', 'neck',
                    'r_sho', 'r_elb', 'r_wri', 'l_sho', 'l_elb', 'l_wri',
                    'r_hip', 'r_knee', 'r_ank', 'l_hip', 'l_knee', 'l_ank',
                    'r_eye', 'l_eye',
                    'r_ear', 'l_ear']
        sigmas = np.array([.26, .79, .79, .72, .62, .79, .72, .62, 1.07, .87, .89, 1.07, .87, .89, .25, .25, .35, .35],
                        dtype=np.float32) / 10.0
        vars = (sigmas * 2) ** 2
        last_id = -1
        color = [0, 224, 255]

        # Take output from network and post-process to desired format
        postprocess_output = inference_output

        stage2_heatmaps = inference_output[-2]
        heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
        heatmaps = cv2.resize(heatmaps, (0, 0), fx=self.upsample_ratio, fy=self.upsample_ratio, interpolation=cv2.INTER_CUBIC)

        stage2_pafs = inference_output[-1]
        pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
        pafs = cv2.resize(pafs, (0, 0), fx=self.upsample_ratio, fy=self.upsample_ratio, interpolation=cv2.INTER_CUBIC)

        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * self.stride / self.upsample_ratio - self.pad[1]) / self.scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * self.stride / self.upsample_ratio - self.pad[0]) / self.scale
        current_poses = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
            pose = Pose(pose_keypoints, pose_entries[n][18])
            current_poses.append(pose)

        if self.track:
            if self.previous_poses != None:
                track_poses(self.previous_poses, current_poses, smooth=self.smooth)
                self.previous_poses = current_poses

        print("")
        print(pose_keypoints)
        print("")
        postprocess_output = pose_keypoints
        return postprocess_output


    def handle(self, data, context):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediciton output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """
       
   
        model_input = self.preprocess(data)
        model_output = self.inference(model_input)
        # print("model_output", model_output)
        output = self.postprocess(model_output)
        # print("output", output)
        # return json.dumps([{"data": "1234"}])
        return [output.tolist()]