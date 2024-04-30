import cv2
import numpy as np
from yoloseg import YOLOSeg

class body_seg_detection():

    def __init__(self):

        self.model_file = "./model/best.onnx"
        self.conf_thres = 0.7
        self.iou_thres = 0.5

    def init_model(self):
        self.yoloseg = YOLOSeg(self.model_file, conf_thres=self.conf_thres, iou_thres=self.iou_thres)

    def run_inference(self,img_data):
        self.boxes, self.scores, self.class_ids, self.masks = self.yoloseg(img_data)
        
    def get_bboxes(self):
        return self.boxes
    
    def get_scores(self):
        return self.scores
    
    def get_masks(self):
        return np.asarray(self.masks,dtype=np.uint8)
    
    def draw_det_res(self,img_data):
        combined_img = self.yoloseg.draw_masks(img_data)
        return combined_img

    def gen_skeletal_frame(self,bbox,mask):

        mask_co_ords = np.asarray(np.where(mask==1))
        spine_pts = []
        spine_pts_y = []

        y_min = np.min(mask_co_ords[0,:])
        y_max = np.max(mask_co_ords[0,:])
        inc_step = round((y_max-y_min)/10)
        spine_pts_y.append(y_min)
        for i in range(9):
            inc_count = 0
            while True:
                cur_y_pt = y_min+(i+1)*inc_step + inc_count
                if cur_y_pt in mask_co_ords[0,:]:
                    spine_pts_y.append(cur_y_pt)
                    break
                inc_count+=1

        spine_pts_y.append(y_max)

        for idx in range(mask_co_ords.shape[1]):
            if(mask_co_ords[0,idx] in spine_pts_y):
                spine_pts.append([mask_co_ords[1,idx],mask_co_ords[0,idx]])

        return spine_pts
    
    def draw_spine(self,spine_pts,img_data):
        
        img_data_dbg = img_data.copy() 
        for i in range(len(spine_pts)-1):
            cv2.line(img_data_dbg,spine_pts[i],spine_pts[i+1],(255,0,0))
        
        return img_data_dbg



if __name__=="__main__":

    
    from time import sleep
    import glob
    import sys

    seg_det = body_seg_detection()
    seg_det.init_model()

    test_data_dir = "./test_data/images_3/"
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
        img = cv2.imread(image_file)
        seg_det.run_inference(img)
        masks = seg_det.get_masks()
        bboxes = seg_det.get_bboxes()

        seg_res_img = seg_det.draw_det_res(img)

        expected_num_spine_pts = 22

        if(len(bboxes) > 0):

            spine_pts_full = seg_det.gen_skeletal_frame(bboxes[0],masks[0])
            spine_pts_full = np.asarray(spine_pts_full)
            y_pts_list = np.unique(spine_pts_full[:,1])
            spine_pts= []
            for y_pt in y_pts_list:
                idxs = np.where(spine_pts_full[:,1] == y_pt)
                x_pts = spine_pts_full[idxs,0]
                spine_pts.append([np.min(x_pts),y_pt])
                spine_pts.append([np.max(x_pts),y_pt])

            if(len(spine_pts) != expected_num_spine_pts):
                print(image_file)
            else:    
                for i in range(len(spine_pts)-1):
                    cv2.line(seg_res_img,spine_pts[i],spine_pts[i+1],(255,0,0))
                np.save(test_data_dir+str(idx)+"_spine.npy",spine_pts)    
                cv2.imshow("Posture anno", seg_res_img)
                key_press = cv2.waitKey(0)
                posture_det = None
                if(key_press == ord('o')):
                    posture_det = "outside"
                elif(key_press == ord('e')):
                    posture_det = "edge"
                elif(key_press == ord('b')):
                    posture_det = "bed_sleep"
                elif(key_press == ord('s')):
                    posture_det = "bed_sit"
                elif(key_press == ord('c')):
                    posture_det = "chair"
                elif(key_press == ord('f')):
                    posture_det = "fall"

            if(posture_det is not None):
                with open(test_data_dir+str(idx)+"_posture.txt",'w') as f:
                    f.write(posture_det)
                    f.close()

        seg_res_img = cv2.resize(seg_res_img, (640,480), interpolation = cv2.INTER_LANCZOS4)
        cv2.imshow("Detected Objects", seg_res_img)
        cv2.waitKey(10)
        
