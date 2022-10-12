import cv2, os

base_path = "./thermo_record/before_preprocessing/"
save_path = "./thermo_record/after_preprocessing/"

imgname_list = os.listdir(base_path)
path_img_list = [base_path+tmp for tmp in imgname_list]

for single_img, single_img_name in zip(path_img_list, imgname_list):
    img = cv2.imread(single_img)
    resize_img = cv2.resize(img, (0, 0), fx=4.0, fy=4.0)
    cv2.imwrite(save_path+single_img_name, resize_img)

#print(f"img.shape = {img.shape}")
#print(f"resize_img.shape = {resize_img.shape}")
#cv2.imshow("img", img)
#cv2.imshow("resize img", resize_img)
#cv2.waitKey()