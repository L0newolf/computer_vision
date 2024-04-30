import cv2
import json
import trt_pose.coco
import trt_pose.models
import torch
import torchvision.transforms as transforms
import PIL.Image
from trt_pose.parse_objects import ParseObjects
from torch2trt import TRTModule
import numpy as np

class DrawObjects(object):

    def __init__(self, topology):
        self.topology = topology

        self.body_labels = {0:'nose', 1: 'left_eye', 2: 'right_eye', 3:'left_ear',
                            4:'right_ear', 5:'left_shoulder', 6:'right_shoulder', 7:'left_elbow',
                            8:'right_elbow', 9:'left_wrist', 10:'right_wrist', 11:'left_hip', 
                            12:'right_hip', 13:'left_knee', 14: 'right_knee', 15: 'left_ankle', 
                            16: 'right_ankle', 17: 'neck'}
        
        self._EDGE_COLORS = [
        (0,255,0), #green (left leg)
        (0,255,0), #green (left leg)
        (0,255,0), #green (right leg)
        (0,255,0), #green (right leg)
        (255,153,204), #light purple (hip)
        (0,0,255), #red (left arm)
        (0,0,255), #red (right arm)
        (0,0,255), #red (left arm)
        (0,0,255), #red (right arm)
        (0,128,255), #orange (forehead betw left right eye)
        (0,128,255), #orange (left cheek)
        (0,128,255), #orange (right cheek)
        (255,255,0), #yellow (left ear)
        (255,255,0), #yellow (right ear)
        (255,255,0), #yellow (left face to shoulder)
        (255,255,0), #yellow (right face to shoulder)
        (51,255,255), #turquoise (nose to neck)
        (255,0,0), #red (left shoulder)
        (255,0,0), #red (right shoulder)
        (204,153,255), #light purple (left side body)
        (204,153,255), #light purple (right side body)
        ]

    def __call__(self, image, object_counts, objects, normalized_peaks):
        topology = self.topology
        height = image.shape[0]
        width = image.shape[1]

        # print(f"counts: {object_counts}")
        count = int(object_counts[0])
        K = topology.shape[0]
        body_list = []
        for i in range(count):
            body_dict = {}
            obj = objects[0][i]
            C = obj.shape[0]
            for j in range(C):
                k = int(obj[j])
                if k >= 0:
                    peak = normalized_peaks[0][j][k]
                    x = round(float(peak[1]) * width)
                    y = round(float(peak[0]) * height)
                    cv2.circle(image, (x, y), 1, (0,255,0) , 2)
                    body_dict[self.body_labels[j]] = (x,y)
            body_list.append(body_dict)
            for k in range(K):
                c_a = topology[k][2]
                c_b = topology[k][3]
                if obj[c_a] >= 0 and obj[c_b] >= 0:
                    peak0 = normalized_peaks[0][c_a][obj[c_a]]
                    peak1 = normalized_peaks[0][c_b][obj[c_b]]
                    x0 = round(float(peak0[1]) * width)
                    y0 = round(float(peak0[0]) * height)
                    x1 = round(float(peak1[1]) * width)
                    y1 = round(float(peak1[0]) * height)
                    cv2.line(image, (x0, y0), (x1, y1), self._EDGE_COLORS[k], 2)
        return body_list, image
    
class key_pt_detection():

    def __init__(self):
        self.model_json_file = "./human_pose.json"
        with open(self.model_json_file, 'r') as f:
            self.human_key_pt = json.load(f)
        self.topology = trt_pose.coco.coco_category_to_topology(self.human_key_pt)
        self.parse_objects = ParseObjects(self.topology, cmap_threshold=0.1, link_threshold=0.1)
        self.draw_objects = DrawObjects(self.topology)
        self.MODEL_WEIGHTS = './densenet121_baseline_att_256x256_B_epoch_160.pth'
        self.OPTIMIZED_MODEL = './densenet121_baseline_att_256x256_B_epoch_160_trt.pth'
        self.model_size = 256

    def init_model(self):
        print("Loading params as Torch tensors to cuda... ")
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
        self.std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
        self.device = torch.device('cuda')
        print("Loading trt model, might take ~30s...")
        self.model = TRTModule()
        self.model.load_state_dict(torch.load(self.OPTIMIZED_MODEL))
        print("TRT Model successfully loaded.")
    
    def preprocess(self,image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = PIL.Image.fromarray(image)
        # image = image.resize((size,size),resample=PIL.Image.BILINEAR)
        image = transforms.functional.to_tensor(image).to(self.device)
        image.sub_(self.mean[:, None, None]).div_(self.std[:, None, None])
        return image[None, ...]

    def key_pt_inference(self,image):
        image_copy = image.copy()
        data = self.preprocess(image)
        cmap, paf = self.model(data)
        cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
        counts, objects, peaks = self.parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
        body_dict, output_image = self.draw_objects(image_copy, counts, objects, peaks)
        return output_image, body_dict
    
    def resize_for_inf(self,image):
        image_resized = cv2.resize(image, dsize=(self.model_size, self.model_size), 
                                   interpolation=cv2.INTER_AREA)
        return image_resized

if __name__=="__main__":

    import glob
    from time import sleep

    key_pt_det = key_pt_detection()
    key_pt_det.init_model()
    test_data_dir = "./test_data/images/"
    images = glob.glob(test_data_dir+'*.jpg')
    anno_images=[]
    anno_images_idx=[]
    for img in images:
        if "frame_" not in img:
            cur_idx = img.split(".")[0]
            cur_idx = cur_idx.split("/")[-1]
            anno_images_idx.append(int(cur_idx))
    anno_images_idx.sort()

    for idx in anno_images_idx:
        image_file = test_data_dir+str(idx)+".jpg"
        print(image_file)
        img = cv2.imread(image_file)
        img_rsz = key_pt_det.resize_for_inf(img)
        output_image, body_key_pts = key_pt_det.key_pt_inference(img_rsz)
        cv2.imshow("body key pt",output_image)
        cv2.waitKey(10)
        sleep(0.25)