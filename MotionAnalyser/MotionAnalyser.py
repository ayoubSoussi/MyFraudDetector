import argparse
import logging
import time

import cv2
import numpy as np

from MotionAnalyser.tf_pose import common
from MotionAnalyser.tf_pose.estimator import TfPoseEstimator
from MotionAnalyser.tf_pose.networks import get_graph_path, model_wh
import math

class MotionAnalyser :
    def __init__(self,service):
        self.myService = service
        self.e = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(432, 368))
        self.human_parts = None
        self.elbows = [(0,0),(0,0)]
        self.shoulders = [(0,0),(0,0)]
        self.neck = (0,0)
    def reinitialize(self):
        self.elbows = [(0, 0), (0, 0)]
        self.shoulders = [(0, 0), (0, 0)]
        self.neck = (0, 0)
    def analyse(self,img):
        humans = self.e.inference(img, resize_to_default=(432 > 0 and 368 > 0), upsample_size=4.0)
        image, parts = TfPoseEstimator.draw_humans(img, humans, imgcopy=False)
        self.human_parts = parts
        for i in range(common.CocoPart.Background.value):
            if i not in parts.keys():
                continue
            body_part = parts[i]
            # print("index :"+str(body_part.part_idx)+"    name : "+str(body_part.get_part_name()))
            if   body_part.part_idx == 1 :
                self.neck = (body_part.x, body_part.y)
            elif body_part.part_idx == 2 :
                self.shoulders[0] = (body_part.x, body_part.y)
            elif body_part.part_idx == 3:
                self.elbows[0] = (body_part.x, body_part.y)
            elif body_part.part_idx == 5 :
                self.shoulders[1] = (body_part.x, body_part.y)
            elif body_part.part_idx == 6 :
                self.elbows[1] = (body_part.x, body_part.y)
        print(self.calculateAngles())
        self.reinitialize()
        return img
    def calculateAngles(self):
        angles = [0.0,0.0]
        if self.neck != (0,0):
            if self.shoulders[0] != (0,0) and self.elbows[0]!=(0,0) :
                angle = self.calculateAngle(self.neck,self.shoulders[0],self.elbows[0])
                # if angle != None :
                angles[0] = angle
                # update the angle in firebase
                self.myService.update_Angle(angle,"r")
            if self.shoulders[1] != (0,0) and self.elbows[1]!=(0,0) :
                angle = self.calculateAngle(self.neck,self.shoulders[1],self.elbows[1])
                # if angle != None :
                angles[1] = angle
                # update the angle in firebase
                self.myService.update_Angle(angle, "l")

        return angles

    def calculateAngle(self,neck,shoulder,elbow):
        x1,y1 = neck
        x2,y2 = shoulder
        x3,y3 = elbow
        if y1<y2 :
            if y2<y3 :
                theta1 = math.degrees(math.atan(abs(y1-y2)/abs(x1-x2)))
                theta2 = math.degrees(math.atan(abs(x2-x3)/abs(y2-y3)))
                return theta1+theta2+90
            elif y2>y3:
                theta1 = math.degrees(math.atan(abs(y1 - y2) / abs(x1 - x2)))
                theta2 = math.degrees(math.atan(abs(y2 - y3) / abs(x2 - x3)))
                return theta1 + theta2 + 180
        elif y1>y2 :
            if y2 < y3:
                theta1 = math.degrees(math.atan(abs(x1 - x2) / abs(y1 - y2)))
                theta2 = math.degrees(math.atan(abs(x2 - x3) / abs(y2 - y3)))
                return theta1 + theta2
            elif y2 > y3:
                theta1 = math.degrees(math.atan(abs(x1 - x2) / abs(y1 - y2)))
                theta2 = math.degrees(math.atan(abs(y2 - y3) / abs(x2 - x3)))
                return theta1 + theta2+90
        return None






# index :0    name : CocoPart.Nose
# index :1    name : CocoPart.Neck
# index :2    name : CocoPart.RShoulder
# index :3    name : CocoPart.RElbow
# index :4    name : CocoPart.RWrist
# index :5    name : CocoPart.LShoulder
# index :6    name : CocoPart.LElbow
# index :7    name : CocoPart.LWrist
# index :8    name : CocoPart.RHip
# index :9    name : CocoPart.RKnee
# index :11    name : CocoPart.LHip
# index :12    name : CocoPart.LKnee
# index :14    name : CocoPart.REye
# index :15    name : CocoPart.LEye
# index :16    name : CocoPart.REar
# index :17    name : CocoPart.LEar