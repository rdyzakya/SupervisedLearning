#KNN
import math
import pandas as pd
import time

class ColumnNumberError(Exception):
	pass

class KNNClassifier:
	def __init__(self,df,x_labels, y_label):
		self.df = df.copy()
		self.x_labels = [self.df.columns[i] for i in x_labels]
		self.y_label = self.df.columns[y_label]

	def euclidean_distance(self,data,other):
		total = 0
		for label in self.x_labels:
			delta = (data[label] - other[label])
			total += delta**2
		return math.sqrt(total)

	#using apply function to reduce time taken
	def predict(self,data,k,print_time=False):
		start = time.time()
		if len(data) != len(self.x_labels):
			raise ColumnNumberError()
		temp = data
		data = {}
		for i in range(len(self.x_labels)):
			data[self.x_labels[i]] = temp[i]


		self.df['distance'] = self.df.apply(lambda row: self.euclidean_distance(data,row[self.x_labels]),axis = 1)

		result = self.df.nsmallest(k,columns = 'distance')[self.y_label].mode()[0]
		time_taken = round(time.time() - start, 4)
		if print_time:
			print("Time taken to predict : " + str(time_taken) + " s")
		return result