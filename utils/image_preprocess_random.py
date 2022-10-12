import cv2
import os
import random
from tqdm import tqdm

base_path = "./record/before_preprocessing/"
save_path = "./record/after_preprocessing/"
model_path = '/home/safetylab/2022_PNUSafetyNet_FallPrediction/AlphaPose/pretrained_models/ESPCN_x4.pb'
picture_num = 100

imgname_list = os.listdir(base_path)
imgname_list = random.sample(imgname_list, picture_num)
path_img_list = [base_path+tmp for tmp in imgname_list]

sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel(model_path)
sr.setModel("espcn", 4)

for single_img, single_img_name in tqdm(zip(path_img_list, imgname_list)):
    img = cv2.imread(single_img)
    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = sr.upsample(img)
    cv2.imwrite(save_path+single_img_name, img)

