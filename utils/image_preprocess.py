import cv2
import os

base_path = "../record/before_preprocessing/"
save_path = "../record/after_preprocessing/"
model_path = '/home/safetylab/AlphaPose/alphapose/utils/ESPCN_x4.pb'

imgname_list = os.listdir(base_path)
path_img_list = [base_path+tmp for tmp in imgname_list]

sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel(model_path)
sr.setModel("espcn", 4)

for single_img, single_img_name in zip(path_img_list, imgname_list):
    img = cv2.imread(single_img)
    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = sr.upsample(img)
    cv2.imwrite(save_path+single_img_name, img)

