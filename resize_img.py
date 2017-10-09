import os,sys
import cv2
dir_path = '/media/weiliu/data/MIOTCD/MIO-TCD-Classification/train/bicycle'
dir_path_new = dir_path + '_resized'
# print os.listdir(dir_path)
# print len(os.listdir(dir_path))
if os.path.exists(dir_path_new):
    print "%s exists" % dir_path_new
else:
    os.mkdir(dir_path_new)
num_imgs = len(os.listdir(dir_path))
index = 1
for filename in os.listdir(dir_path):
    img_full_name = os.path.join(dir_path, filename)
    img_name_tosave = "{}.jpg".format(str(index).zfill(len(str(num_imgs))))
    if index < 10:
        print img_name_tosave
        print img_full_name
    resized_image = cv2.resize(cv2.imread(img_full_name), (64, 64))
    cv2.imwrite(os.path.join(dir_path_new, img_name_tosave),resized_image)
    index = index + 1
