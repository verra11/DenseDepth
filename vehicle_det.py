import numpy as np
import argparse
import imutils
import cv2
import os

classesFile = "coco.names"
classNames = []
with open(classesFile, "rt") as f:
	classNames = f.read().rstrip('\n').split('\n')

labels = classNames
# labels = ["bicycle", "car", "motorbike", "bus", "truck"]

modelConfig = "yolov3-320.cfg"
modelWeights = "yolov3.weights"


NMSThresh = 0.4
confThresh = 0.85

net = cv2.dnn.readNetFromDarknet(modelConfig, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def findVehicles(outputs, img):

	hT, wT, cT = img.shape

	orig = img.copy()

	bbox = []
	confs = []
	classIDs = []

	for output in outputs:
		for det in output:
			scores = det[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			if confidence > confThresh and (classNames[classID] in labels):

				w, h = int(det[2]*wT), int(det[3]*hT)
				x, y = int((det[0]*wT) - w/2), int((det[1]*hT) - h/2)

				bbox.append([x, y, w, h])
				confs.append(float(confidence))
				classIDs.append(classID)

	indices = cv2.dnn.NMSBoxes(bbox, confs, confThresh, NMSThresh)

	boxes = []

	for i in indices:
		i = i[0]

		x, y, w, h = bbox[i]

		boxes.append((int(x-w/2), int(y-h/2), int(x+w), int(y+h)))

		cv2.rectangle(img, (int(x-w/2),int(y-h/2)), (int(x+w), int(y+h)), 
			(0, 0, 255), 4)
		cv2.putText(img, str(classNames[classIDs[i]]).upper() + str(": {:.2f}".format(confs[i] * 100)),
			(int(x+w/3), y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

	return boxes

layerNames = net.getLayerNames()
outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()] 


def detectVehicle(img):
	# img = cv2.resize(img, (640, 480))
	img = imutils.resize(img, height=512)
	# img = cv2.GaussianBlur(img, (5, 5), 0)
	blob = cv2.dnn.blobFromImage(img, 1/255, (320, 320), [0, 0, 0], 1, crop=False)
	net.setInput(blob)

	outputs = net.forward(outputNames)

	boxes = findVehicles(outputs, img)

	return boxes
