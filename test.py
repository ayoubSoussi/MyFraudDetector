import copy
import time

import cv2

from imutils.video import VideoStream

from MotionAnalyser.MotionAnalyser import MotionAnalyser
from MyService.myService import MyService
from ObjectDetector.ObjectDetector import Detector
from Recognizer.Recognizer import Recognizer









# initialiser le video stream et pointer sur la sortie du fichier video, puis
# permettre la caméra de se réchauffer

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# boucler sur des frames du video stream
myservice = MyService(217,1)
recognizer = Recognizer(myservice)
motionAnalyser = MotionAnalyser(myservice)
objectDetector = Detector(myservice)
while  True:

    # récuperer un frame du video stream
    origin_frame = vs.read()
    detected_frame = objectDetector.detectObjects(copy.deepcopy(origin_frame))
    recognized_frame = recognizer.addImage(copy.deepcopy(origin_frame))
    motion_analyzed_frame = motionAnalyser.analyse(copy.deepcopy(origin_frame))


    # afficher le streaming
    cv2.imshow("Frame", detected_frame)
    key = cv2.waitKey(1) & 0xFF

# arrêter le video stream
cv2.destroyAllWindows()
vs.stop()

# check to see if the video writer point needs to be released
if writer is not None:
    writer.release()
time.sleep(2.0)
