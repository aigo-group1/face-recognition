import cv2 as cv2
import numpy as np
from PIL import Image
from yolo import YOLO
from yolo import detect_video
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import os
import argparse
import sys

parse = argparse.ArgumentParser(description='Process parameters yolo')
parse.add_argument('--video',type=str,default='',help='Your Video Path')
args = parse.parse_args()

BASE_PATH  = os.path.join(os.getcwd(),'face-detection')
MODEL_DATA = os.path.join(BASE_PATH,'model_data')
MODEL_PATH = os.path.join(MODEL_DATA,'face_human.h5')
ANCHORS_PATH = os.path.join(MODEL_DATA,'face_anchors.txt')
CLASSES_PATH = os.path.join(MODEL_DATA,'face_classes.txt')
option = {
        "model_path": MODEL_PATH,
        "anchors_path": ANCHORS_PATH,
        "classes_path": CLASSES_PATH,
        "score" : 0.25,
        "iou" : 0.35,
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
}
yolo_model = YOLO(**option)
#lay ra 1 list câc dictionary, mỗi dictionary gồm label và box, trong đó label chứa class name(face hoặc human ở đây mình chỉ chọn face thôi)
#còn box chứa tọa độ 2 điểm top left và right bottom
def getFrameBoundingBoxes(frame):
    img_to_detect = Image.fromarray(frame)
    img_detect,data = yolo_model.detect_image(img_to_detect)
    return data
#detect trên từng frame hình, trả về hình ảnh frame đó đã được detect hết mặt người nếu có
def detect_frame(frame):
    img_to_detect = Image.fromarray(frame)
    img_detect,data = yolo_model.detect_image(img_to_detect)
    for item in data:
        if 'face' in item['label']:
            x1,y1,x2,y2 = item['box']
            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),thickness=5)
    return frame
#detect trên video, các thâm số là đường dẫn tới video và isout mặc định là True nếu muốn xuất ra video đầu ra
def detect_video(video_path,isout=True):
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = isout if isout == True else False
    if isOutput:
        out = cv2.VideoWriter("output3.avi",cv2.VideoWriter_fourcc(*"MJPG"), video_fps,video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    count = 0
    while True:
        count+=1
        return_value, frame = vid.read()
        if frame is not None:
            image = detect_frame(frame)
            result = image
            curr_time = timer()
            exec_time = curr_time - prev_time
            prev_time = curr_time
            accum_time = accum_time + exec_time
            curr_fps = curr_fps + 1
            if accum_time > 1:
                accum_time = accum_time - 1
                fps = "FPS: " + str(curr_fps)
                curr_fps = 0
            cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.50, color=(255, 0, 0), thickness=2)
            cv2.namedWindow("result", cv2.WINDOW_NORMAL)
            cv2.imshow("result", result)
            if isOutput:
                out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    yolo_model.close_session()

if __name__ =='__main__':
    video_path = args.video
    print(video_path)
    print(BASE_PATH)
    print(MODEL_PATH)
    detect_video(video_path)