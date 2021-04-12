import numpy as np
import matplotlib
import argparse
import time
import glob
import cv2
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'

from tensorflow.keras.models import load_model
from layers import BilinearUpSampling2D
from utils import predict, load_images, display_images
from vehicle_det import detectVehicle
from centroidtracker import CentroidTracker
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to video file")
ap.add_argument("-m", "--model", default="kitti.h5",
	help="path to saved model")
args = vars(ap.parse_args())

custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

print("[INFO] Loading Model...")
model = load_model(args["model"], custom_objects = custom_objects, compile=False)

ct = CentroidTracker()

if args["video"] is not None:
	vs = cv2.VideoCapture(args["video"])
else:
	vs = cv2.VideoCapture(0)

cnt = 0


name = args["video"].split("/")[-1].split(".")[0]
dirName = name + " speed frames"
# os.mkdir(dirName)

while True:

	ret, frame = vs.read()

	if ret == False:
		break

	cnt += 1
	frame = cv2.resize(frame, (640, 480))
	# frame = cv2.GaussianBlur(frame, (3, 3), 0)
	inp = np.clip(np.asarray(frame) / 255 , 0, 1)
	inp = np.expand_dims(inp, 0)

	output = predict(model, inp)
	output = np.zeros((240, 320))
	output = np.expand_dims(output, 0)
	output = np.expand_dims(output, -1)
	output = output*80
	# output = 1 / output

	boxes = detectVehicle(frame)
	img = frame.copy()

	for (x, y, x2, y2) in boxes:
		cv2.rectangle(img, (x, y), (x2, y2), (0, 255, 0), 2)

	objects = ct.update(boxes, output.copy(), cnt)

	for (objectID, centroid) in objects.items():

		dX = int(centroid[0] // 2.0)
		dY = int(centroid[1] // 2.0)

		# print("[INFO] centroid {}: Original: ({},{}) Scaled: ({},{})".format(objectID, centroid[0], centroid[1],
		# 	dX, dY)) 

		depth = None
		if dY<240 and dX<340:
			depth = output[0][dY][dX][0]

		# if depth is not None:
		# 	print("[INFO] Depth of ID-{} : {:.2f}m".format(objectID, depth))
		# 	if ct.depth[objectID][0] is not None and ct.depth[objectID][1] is not None:
		# 		print("[INFO] Change in Depth of ID-{} : {:.2f}m".format(objectID, np.abs(ct.depth[objectID][1]-ct.depth[objectID][0])))
		# print("[INFO] Frame of ID-{} : {}, {}".format(objectID, ct.frame[objectID][0], ct.frame[objectID][1]))
		if ct.speed[objectID] is not None:
			print("[INFO] Speed of ID-{} : {:.5f} kmph".format(objectID, ct.speed[objectID]))

		speed = ct.speed[objectID]
		# speed = 20. + np.random.rand()*(50. - 20.)
		if speed is None:
			text = "N/A"
		else:
			text = "{:.2f} kmph".format(speed)

		cv2.putText(img, text, (ct.box[objectID][0]-5, ct.box[objectID][1]-5),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
		cv2.circle(img, (centroid[0], centroid[1]), 2, (0, 255, 0), -1)

	# frame = cv2.resize(frame, (320, 240))
	# img = cv2.resize(img, (320, 240))

	# viz = display_images(output.copy(), inp.copy())

	# plt.figure(figsize=(10, 4))
	# plt.imshow(viz)
	# plt.show()

	# output = output[0]
	# output = cv2.merge([output, output, output])
	viz = np.hstack([frame, img])

	cv2.imshow("IMG", viz)
	# cv2.imshow("IMG2", output)

	# fileName = dirName + "/" + name + str(cnt) + ".png"
	# if (cnt%20) == 0:
	# 	cv2.imwrite(fileName, img)

	if cv2.waitKey(1) & 0xff == ord('q'):
		break


if args["video"] is not None:
	vs.release()
else:
	vs.release()

cv2.destroyAllWindows()