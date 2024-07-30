from .camera_geometry import CameraGeometry
import numpy as np
import albumentations as albu
import cv2
import torch
import segmentation_models_pytorch as smp
from gym_carlaRL.envs.lanenet_lane_detection_pytorch.model.lanenet.LaneNet import LaneNet
import matplotlib.pyplot as plt


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class LaneDetector():
    #def __init__(self, cam_geom=CameraGeometry(image_width=512, image_height=256), model_path='./lane_segmentation_model.pth',
    #encoder = 'efficientnet-b0', encoder_weights = 'imagenet'):
    def __init__(self, model_path, cam_geom=CameraGeometry(image_width=512, image_height=256),
    encoder = 'efficientnet-b0', encoder_weights = 'imagenet', intersection=False):
        self.cg = cam_geom
        self.cut_v, self.grid = self.cg.precompute_grid()
        self.model = LaneNet()  # Instantiate the model architecture
        state_dict = torch.load(model_path)
        if intersection:
            state_dict = torch.load(model_path)['model_state_dict']
        else:
            state_dict = torch.load(model_path)
        self.model.load_state_dict(state_dict)  # Apply the state dictionary
        self.model.eval()
        self.model.to(DEVICE)
        self.encoder = encoder
        self.encoder_weights = encoder_weights
        preprocessing_fn = smp.encoders.get_preprocessing_fn(self.encoder, self.encoder_weights)
        self.to_tensor_func = self._get_preprocessing(preprocessing_fn)

    def _get_preprocessing(self, preprocessing_fn):
        def to_tensor(x, **kwargs):
            return x.transpose(2, 0, 1).astype('float32')
        transform = [
            albu.Lambda(image=preprocessing_fn),
            albu.Lambda(image=to_tensor),
        ]
        return albu.Compose(transform)

    
    def read_imagefile_to_array(self, filename):
        image = cv2.imread(filename)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    

    def detect_from_file(self, filename):
        img_array = self.read_imagefile_to_array(filename)
        return self.detect(img_array)


    def detect(self, img_array):
        #image_tensor = self.to_tensor_func(image=img_array)["image"]
        #x_tensor = torch.from_numpy(image_tensor).to(self.device).unsqueeze(0)
        #model_output = self.model.forward(x_tensor).cpu().numpy()
        with torch.no_grad():
            model_output = self.model.forward(img_array)

        segmentation_map = torch.squeeze(model_output['binary_seg_pred']).to('cpu').numpy() * 255
        segmentation_map = segmentation_map.astype(np.uint8)

        # Display the segmentation map
        cv2.imshow('Segmentation', segmentation_map)
        cv2.waitKey(1)
        #print(model_output.shape)
        #background, left, right = model_output[0,0,:,:], model_output[0,1,:,:], model_output[0,2,:,:]
        # Assuming model_output is your output tensor


        #segmentation_map = model_output[0, 0, :, :]* 255
        # If needed, apply thresholding to create a binary segmentation map
        # Example: segmentation_map = (segmentation_map > threshold_value).astype(np.uint8)

        # Initialize masks for left and right lanes
        left_lane_mask = np.zeros_like(segmentation_map)
        right_lane_mask = np.zeros_like(segmentation_map)

        # Define criteria for distinguishing left and right lanes
        # This could be based on the distribution of lane pixels across the width of the image
        # For simplicity, let's use the median x-coordinate of lane pixels
        lane_pixels = np.argwhere(segmentation_map > 0)  # Assuming non-zero pixels are lane pixels
        if len(lane_pixels) == 0:
            print("no lanes detected")
        if len(lane_pixels) > 0:
            median_x_coordinate = np.median(lane_pixels[:, 1])  # x-coordinate is the second value

            for y, x in lane_pixels:
                if x < median_x_coordinate:
                    left_lane_mask[y, x] = 1
                else:
                    right_lane_mask[y, x] = 1
        return 0, left_lane_mask, right_lane_mask, segmentation_map
        #return background, left, right
    
    def detect_and_fit(self, img_array):
        with torch.no_grad():
            model_output = self.model(img_array)

        segmentation_map = torch.squeeze(model_output['binary_seg_pred']).to('cpu').numpy()
        segmentation_map = segmentation_map.astype(np.uint8)

        # # Display the segmentation map
        # cv2.imshow('Segmentation', segmentation_map)
        # cv2.waitKey(1)
        
        return segmentation_map

    def fit_poly(self, probs):
        probs_flat = np.ravel(probs[self.cut_v:, :])
        mask = probs_flat > 0.3

        if not np.any(mask):
            # Handle the empty case (e.g., return a default polynomial or None)
            return np.poly1d([0])  # Example: returning a polynomial that always evaluates to 0

        coeffs = np.polyfit(self.grid[:, 0][mask], self.grid[:, 1][mask], deg=3, w=probs_flat[mask])
        return np.poly1d(coeffs)

        #probs_flat = np.ravel(probs[self.cut_v:, :])
        #mask = probs_flat > 0.3
        #coeffs = np.polyfit(self.grid[:,0][mask], self.grid[:,1][mask], deg=3, w=probs_flat[mask])
        #return np.poly1d(coeffs)

    def __call__(self, img):
        if isinstance(img, str):
            img = self.read_imagefile_to_array(img)
            #cv2.namedWindow("Display window", cv2.WINDOW_NORMAL)
            # Display the image
            #cv2.imshow("Display window", img)

        return self.detect_and_fit(img)
    