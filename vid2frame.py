import argparse
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
	help="path to the video file")

args = vars(ap.parse_args())

name = args["video"].split("/")[-1].split('.')[0]
vs = cv2.VideoCapture(args["video"])
cnt = 0

dirName = name+" frames" 
os.mkdir(dirName)

while True:

	ret, frame = vs.read()

	cnt += 1

	if (cnt%50) != 0:
		continue

	if ret == False:
		break

	fileName = dirName + '/' + name + str(cnt) + ".png"

	frame = cv2.resize(frame, (640, 480))
	cv2.imwrite(fileName, frame)

print("Total frames: {}".format(cnt))

vs.release()
cv2.destroyAllWindows()