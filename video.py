from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util import *
import argparse
import os
import os.path as osp
from darknet import Darknet
import pickle as pkl
import pandas as pd
import random
from collections import defaultdict
from collections import Counter


class_name = "bottle"


def arg_parse():
    """
    Parse arguments to the detect module

    """

    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
    parser.add_argument("--bs", dest="bs", help="Batch size", default=1)
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions",
                        default=0.5)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshold", default=0.4)
    parser.add_argument("--cfg", dest='cfgfile',
                        help="Config file",
                        default="cfg/yolov3.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help="weightsfile",
                        default="yolov3.weights", type=str)
    parser.add_argument("--reso", dest='reso', help="Input resolution of the network. Increase to increase accuracy. "
                                                     "Decrease to increase speed",
                        default="416", type=str)
    parser.add_argument("--video", dest="videofile", help="Video file to run detection on", default="video.avi",
                        type=str)

    return parser.parse_args()


args = arg_parse()
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thresh = float(args.nms_thresh)
start = 0
CUDA = torch.cuda.is_available()

num_classes = 80
classes = load_classes("data/coco.names")

# Set up the neural network
print("Loading network.....")
model = Darknet(args.cfgfile)
model.load_weights(args.weightsfile)
print("Network successfully loaded")

model.net_info["height"] = args.reso
inp_dim = int(model.net_info["height"])
assert inp_dim % 32 == 0
assert inp_dim > 32

# If there's a GPU available, put the model on GPU
if CUDA:
    model.cuda()

# Set the model in evaluation mode
model.eval()


def write(x, results):
    c1 = tuple(map(int, x[1:3].int()))
    c2 = tuple(map(int, x[3:5].int()))
    img = results
    cls = int(x[-1])
    color = random.choice(colors)
    label = "{0}".format(classes[cls])
    cv2.rectangle(img, c1, c2, color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, color, -1)

    #Extracting the region of interest(ROI) for the detected object
    roi = img[c1[1]:c2[1], c1[0]:c2[0]]

    #Convert the ROI to HSV color space
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    #Calculate the dominant color in ROI
    pixels = hsv_roi.reshape(-1,3)
    color_counts = Counter(tuple(pixels) for pixel in pixels)
    dominant_color = color_counts.most_common(1)[0][0]

    #Get the object class name
    class_name = classes[cls]

    #Set the color based on the object class
    class_colors = {
        "bottle": (0,0,255), #Red colour for bottles
        "person": (0,255,0)  #Green color for people
    }
    color = class_colors.get(class_name, (0,0,0))

    #Draw the label with the object class and dominant color
    label = f"{class_name}  ({dominant_color})"
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)

    #Update the color count with the dominant color of the detected object
    color_counts[dominant_color] +=1
    #return img


def get_test_input(input_dim, CUDA):
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (input_dim, input_dim))
    img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    if CUDA:
        img_ = img_.cuda()

    return img_




#ip_camera_url = "http://192.168.63.153:8080/video"
# Detection phase
cap = cv2.VideoCapture(0)  # for usb webcam

if not cap.isOpened():
    print("Failed to open IP camera stream")
    exit()

frames = 0
start = time.time()

color_count = Counter()
colors = pkl.load(open("pallete", "rb"))

object_count = 0
prev_count = 0
total_count = 0

total_object_count = defaultdict(int)

sum_2 =0
check = False

sum_total=0
main_cnt =0
temp=0
temp_p=0

objects_in_frame_prev = 0
while cap.isOpened():
    h = 100
    w = 100
    frame2 = h*w
    
    ret, frame2 = cap.read()
    
    if ret:
        img = prep_image(frame2, inp_dim)
        im_dim = frame2.shape[1], frame2.shape[0]
        im_dim = torch.FloatTensor(im_dim).repeat(1, 2)

        if CUDA:
            im_dim = im_dim.cuda()
            img = img.cuda()

        with torch.no_grad():
            output = model(Variable(img), CUDA)
        output = write_results(output, confidence, num_classes, nms_conf=nms_thresh)

        if type(output) == int:
            frames += 1
            print("FPS of the video is {:5.4f}".format(frames / (time.time() - start)))
            cv2.imshow("frame", frame2)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            continue

        im_dim = im_dim.repeat(output.size(0), 1)
        scaling_factor = torch.min(416 / im_dim, 1)[0].view(-1, 1)

        output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim[:, 0].view(-1, 1)) / 2
        output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim[:, 1].view(-1, 1)) / 2

        output[:, 1:5] /= scaling_factor

        for i in range(output.shape[0]):
            output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim[i, 0])
            output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim[i, 1])
        
        # Count the number of objects
        # Count the number of objects
        object_count = 0
        objects_in_frame = {}
        main_cnt =0
   
        #print("AA")
        for i in range(output.shape[0]):
            class_index = int(output[i, -1])
            if classes[class_index] == class_name:
                bbox = output[i, 1:5].cpu().detach().numpy()
                bbox = [int(x) for x in bbox]
                objects_in_frame[tuple(bbox)] = True                            #The coordinates of the bounding box are stored as a tuple in the dictionary
                
        #print("OBJECT IN FRAME:"+str(len(objects_in_frame)))
        #print("OBJECT IN PREV FRAME:"+str(objects_in_frame_prev))
        
        if objects_in_frame_prev == len(objects_in_frame):
            check = False
        else:
            check =True
            objects_in_frame_prev = len(objects_in_frame)

        # Check if any previously counted objects are still in the frame
        objects_to_remove = []
        if class_name in total_object_count:
            for obj in total_object_count[class_name]:
                if obj not in objects_in_frame:
                    objects_to_remove.append(obj)
                    main_cnt += len(objects_to_remove)                          #The objects which have been detected but aren't in the frame are added to the main_cnt
                    
  
                    #print(main_cnt)
                    #sum_total += main_cnt

        # Count the objects that are newly detected and not in the previous count
        new_objects = [obj for obj in objects_in_frame if obj not in total_object_count.get(class_name, {})]
        new_objects += objects_to_remove

        #Update the object count and the total object count
        sum_2 += len(new_objects)
        object_count += len(new_objects)
        total_object_count[class_name] = objects_in_frame 
        #print(object_count)

        # Remove the objects that are still in the frame from the previous count
        if class_name in total_object_count:
            for obj in objects_to_remove:
                total_object_count[class_name].pop(obj, None)
                main_cnt += len(total_object_count.get(class_name, {}))
                #check = False


        #sum_total += object_count 
        sum_total = len(total_object_count.get(class_name, {})) 
        sum_2 = sum_total 
        temp_p= sum_2
        #main_cnt = len(objects_to_remove) + len(total_object_count.get(class_name, {}))
        list(map(lambda x: write(x, frame2), output))

        cv2.putText(frame2, f"Total {class_name} objects until now: {temp}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)

        cv2.imshow("frame", frame2)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

        frames += 1
       # print("FPS of the video is {:5.2f}".format(frames / (time.time() - start)))

    
    else:
        break
    
    if check:
        temp += objects_in_frame_prev 
        check = False

    main_cnt=0
    #print("count:")
    #print(temp)
    