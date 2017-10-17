from darkflow.net.build import TFNet
import cv2

# flow --model cfg/yolo-new.cfg --train --dataset "~/VOCdevkit/VOC2007/JPEGImages" --annotation "~/VOCdevkit/VOC2007/Annotations"
options = {"model": "cfg/yolo.cfg",
           "train" : "true",
           "dataset" : "../Total_Data/VOCdevkit/VOC2007/JPEGImages",
           "annotation" : "../Total_Data/VOCdevkit/VOC2007/Annotations"}
tfnet = TFNet(options)
tfnet.train()