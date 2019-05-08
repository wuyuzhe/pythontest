#-*- coding:utf-8 -*-
import cv2
import dblib
import face_recognition

from subprocess import call
from time import time

FREQ = 5
<<<<<<< HEAD
FACE_DETECTOR = dblib.get_frontal_face_detector()
=======
FACE_DETECTOR = dlib.get_frontal_face_detector()
>>>>>>> cf2f463e0d48f2a55c32beff341ca1b4b5c9e5ac

def notify(text,tilte):
	cmd = r'display notification "%s" with tilte "%s"'%(text,title)
	call(["oscascript" , "-e" , cmd])
if __name__ == '__main__':
	#初始化摄像头
	cap = cv2.VideoCapture(0)
	#绘图窗口
	cv2.namedWindow('face')
	notify_time = 0
	while True:
		ret,frame = cap.read()
		frame = cv2.resize(frame , (320,240))
		faces = FACE_DETECTOR(frame , 1)
		
	for face in faces:
		fimg = frame[face.top():face.bottom(),face.left():face.right()]
		cv2.rectangle(frame,(face.left(),face.top()),(face.right(),face.bottom()),(255,0,0) , 3)
	if time() - notify_time>FREQ:
		notify(u'检测到人脸',u'注意')
		notify_time = time()
	#画到窗口里
	cv2.imshow('face',frame)
	cv2.destroyAllWindows()
	cap.release()

# def recognition(f1 , f2):
# 	known_image = face_recognition.load_image_file(f1)
# 	unknown_image = face_recognition.load_image_file(f2)

# 	biden_encoding = face_recoginition.face_encodings(known_image)[0]
# 	unknown_encoding = face_recoginition.face_encodings(unknown_image)[0]

# 	results = face_recoginition.compare_faces([biden_encoding] , unknown_encoding)
# 	return reuslts
