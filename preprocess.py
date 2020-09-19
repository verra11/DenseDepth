import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,
	help="path to input images")
ap.add_argument("-o", "--output", default="./out.png",
	help="path to output image file")
args = vars(ap.parse_args())

img = cv2.imread(args["image"])
img = cv2.resize(img, (640, 480))
cv2.imwrite(args["output"], img)