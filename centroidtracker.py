from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import time

class CentroidTracker():

	def __init__(self, maxDisappeared=30):
		self.nextObjectID = 0
		self.objects = OrderedDict()
		self.disappeared = OrderedDict()
		self.depth = OrderedDict()
		self.frame = OrderedDict()
		self.speed = OrderedDict()
		self.box = OrderedDict()

		self.maxDisappeared = maxDisappeared

	def register(self, centroid, depthVal, frameCnt, rect):
		self.objects[self.nextObjectID] = centroid
		self.disappeared[self.nextObjectID] = 0
		self.depth[self.nextObjectID] = [depthVal, depthVal]
		self.frame[self.nextObjectID] = [frameCnt, frameCnt]
		self.speed[self.nextObjectID] = None
		self.box[self.nextObjectID] = rect

		self.nextObjectID += 1

	def deregister(self, objectID):
		del self.objects[objectID]
		del self.disappeared[objectID]
		del self.depth[objectID]
		del self.frame[objectID]
		del self.speed[objectID]
		del self.box[objectID]

	def update(self, rects, depthMap, frameCnt):

		if len(rects) == 0:

			for objectID in list(self.disappeared.keys()):
				self.disappeared[objectID] += 1

				if self.disappeared[objectID] > self.maxDisappeared:
					self.deregister(objectID)

			return self.objects

		inputCentroids = np.zeros((len(rects), 2), dtype="int")

		for (i, (startX, startY, endX, endY)) in enumerate(rects):

			cX = int((startX + endX)/2.0)
			cY = int((startY + endY)/2.0)

			inputCentroids[i] = (cX, cY)

		if len(self.objects) == 0:
			for i in range(0, len(inputCentroids)):
				d = None
				if inputCentroids[i][1]<240 and inputCentroids[i][0]<320:
					d = depthMap[0][inputCentroids[i][1]][inputCentroids[i][0]][0]
				self.register(inputCentroids[i], d, frameCnt, rects[i])

		else:

			objectIDs = list(self.objects.keys())
			objectCentroids = list(self.objects.values())

			D = dist.cdist(np.array(objectCentroids), inputCentroids)

			rows = D.min(axis=1).argsort()

			cols = D.argmin(axis=1)[rows]

			usedRows = set()
			usedCols = set()

			for (row, col) in zip(rows, cols):

				if row in usedRows or col in usedCols:
					continue

				objectID = objectIDs[row]
				self.objects[objectID] = inputCentroids[col]
				self.disappeared[objectID] = 0
				self.frame[objectID][1] = frameCnt
				self.box[objectID] = rects[col]
				if inputCentroids[col][1]<240 and inputCentroids[col][0]<320:
					self.depth[objectID][1] = depthMap[0][inputCentroids[col][1]][inputCentroids[col][0]][0]
					if self.depth[objectID][0] is None:
						self.depth[objectID][0] = depthMap[0][inputCentroids[col][1]][inputCentroids[col][0]][0]
					self.speed[objectID] = float((np.abs(self.depth[objectID][0] - self.depth[objectID][1]) * 60 * 60 * 30) / (np.abs(self.frame[objectID][1] - self.frame[objectID][0])))# * 1000))

				usedRows.add(row)
				usedCols.add(col)

			unusedRows = set(range(0, D.shape[0])).difference(usedRows)
			unusedCols = set(range(0, D.shape[1])).difference(usedCols)

			if D.shape[0]>=D.shape[1]:

				for row in unusedRows:

					objectID = objectIDs[row]
					self.disappeared[objectID] += 1

					if self.disappeared[objectID] > self.maxDisappeared:
						self.deregister(objectID)

			else:

				for col in unusedCols:
					d = None
					if inputCentroids[col][1]<240 and inputCentroids[col][0]<320:
						d = depthMap[0][inputCentroids[col][1]][inputCentroids[col][0]][0]
					self.register(inputCentroids[col], d, frameCnt, rects[col])

		return self.objects