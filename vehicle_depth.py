import matplotlib
import argparse
import glob
import numpy as np
import cv2
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'

from tensorflow.keras.models import load_model
from layers import BilinearUpSampling2D
from utils import predict, load_images, display_images
from vehicle_det import detectVehicle
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="Path to input image")
ap.add_argument("-m", "--model", default="kitti.h5", 
	help="Path to saved model file")
args = vars(ap.parse_args())

custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

print("[INFO] Loading Model...")
model = load_model(args["model"], custom_objects = custom_objects, compile=False)

print("[INFO] Loading images...")
inputs = load_images(glob.glob(args["image"]))

outputs = predict(model, inputs)
outputs *= 80

image = cv2.imread(args["image"])
boxes = detectVehicle(image)

viz = display_images(outputs.copy(), inputs.copy())
plt.figure(figsize=(10, 4))
plt.imshow(viz)
plt.savefig('depthmap-1.png')
plt.show()

# for (inp, out) in zip(inputs, outputs):

# 	# out = cv2.merge([out, out, out])
# 	print(out.shape)
# 	print(inp.shape)
# 	mean = np.mean(out)
# 	print("[INFO] Depth/Distance of Vehicle :" + str(float(mean)))
# 	out = cv2.resize(out, (640, 480))

# 	for b in boxes:
# 		x, y, w, h = b

# 		img1 = np.array(inp[y:y+h, x:x+w])
# 		img2 = np.array(out[y:y+h, x:x+w])

# 		img1 = np.expand_dims(img1, 0)
# 		img2 = np.expand_dims(img2, 0)
# 		img2 = np.expand_dims(img2, -1)

# 		# print(img1.shape)
# 		# print(img2.shape)

# 		mean = np.mean(img2)
# 		print("[INFO] Depth/Distance of Vehicle :" + str(float(mean)))

# 		viz = display_images(img2.copy(), img1.copy())
# 		plt.figure(figsize=(10, 4))
# 		plt.imshow(viz)
# 		plt.show()
