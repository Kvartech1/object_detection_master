#Original total object counter
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


def assign_unique_ids(output, detections, id_dict):
    current_time = time.strftime("%H:%M:%S", time.localtime())
    for i in range(output.shape[0]):
        class_index = int(output[i, -1])
        if classes[class_index] == class_name:
            bbox = output[i, 1:5].cpu().detach().numpy()
            bbox = [int(x) for x in bbox]
            if tuple(bbox) not in detections:
                existing_id = id_dict.get(tuple(bbox))
                if existing_id is None:
                    new_id = current_time
                    id_dict[tuple(bbox)] = new_id
                    detections[tuple(bbox)] = {
                        'id': new_id,
                        'count': 1
                    }
                    print(f"Assigned ID {new_id} to the object")
                else:
                    detections[tuple(bbox)] = {
                        'id': existing_id,
                        'count': 1
                    }
                    print(f"Object already detected with ID {existing_id}")
            else:
                detections[tuple(bbox)]['count'] += 1


def total_count(detections, threshold):
    current_time = time.strftime("%H:%M:%S", time.localtime())
    objects_to_remove = []

    for bbox, data in detections.items():
        unique_id = data['id']
        count = data['count']
        time_difference = int(current_time.split(':')[2]) - int(unique_id.split(':')[2])

        if time_difference > threshold:
            data['count'] += 1
            objects_to_remove.append(bbox)

    #Remove the objects from detections dictionary
    for bbox in objects_to_remove:
        del detections[bbox]

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
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
    return img

ip_camera_url = "http://192.168.63.153:8080/video"
# Detection phase
cap = cv2.VideoCapture(0)  # for usb webcam

if not cap.isOpened():
    print("Failed to open IP camera stream")
    exit()

frames = 0
start = time.time()

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

detections = {}
id_dict = {}
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
        

        #Assigning unique ids to the detected objects

        assign_unique_ids(output, detections, id_dict)

        for obj in detections.values():
            if obj['count'] > 1:
                total_object_count[obj['id']] += 1

        #Removing objects that are no longer present in the current frame
        removed_objects = []
        for bbox in detections.keys():
            if tuple(bbox) not in [tuple(box[1:5].tolist()) for box in output]:
                removed_objects.append(bbox)
        for bbox in removed_objects:
            del detections[bbox]

        list(map(lambda x: write(x, frame2), output))

        cv2.putText(frame2, f"Total {class_name} objects until now: {sum(total_object_count.values())}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)

        cv2.imshow("frame", frame2)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

        frames += 1
       # print("FPS of the video is {:5.2f}".format(frames / (time.time() - start)))