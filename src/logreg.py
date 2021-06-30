#logistic
import math
import pandas as pd
import numpy as np
import time

def sigmoid(x):
	try:
		denominator = 1 + 1/(math.e**x) #1 + e^-x
		return 1/denominator
	except ZeroDivisionError:
		return 0
	except OverflowError:
		return 1

class NotBinaryClassificationProblemError(Exception):
	pass

class LengthNotSameError(Exception):
	pass

def dot(X1,X2):
	if len(X1) != len(X2):
		raise LengthNotSameError()
	res = 0
	for i in range(len(X1)):
		res += X1[i]*X2[i]
	return res

class LogisticRegression:
	def __init__(self):
		self.b_values = None
		self.y_vals = None

	def train(self,df_,x_labels, y_label, alpha, epochs):
		df = df_.copy()
		x_labels = [df.columns[i] for i in x_labels]
		y_label = df.columns[y_label]
		self.y_vals = df[y_label].unique()
		if len(self.y_vals) != 2:
			raise NotBinaryClassificationProblemError()
		else:
			if 1 not in self.y_vals or 0 not in self.y_vals:
				df.loc[df[y_label] == self.y_vals[0], y_label] = 0
				df.loc[df[y_label] == self.y_vals[1], y_label] = 1

		b_values = [0 for i in range(1 + len(x_labels))]

		for epoch in range(epochs):
			start = time.time()
			#using iterrows to reduce loop time
			for idx, row in df.iterrows():
				row_vector = [1] + [row[j] for j in x_labels]
				z = dot(row_vector,b_values)
				prediction = sigmoid(z)
				b_values = [b_values[i] + alpha*(row[y_label] - prediction)*prediction*(1 - prediction)*row_vector[i] for i in range(len(b_values))]

			naccurate = 0
			for idx,row in df.iterrows():
				row_vector = [1] + [row[j] for j in x_labels]
				z = dot(row_vector,b_values)
				prediction = sigmoid(z)
				if prediction < 0.5 and row[y_label] == 0:
					naccurate += 1
				if prediction >= 0.5 and row[y_label] == 1:
					naccurate += 1
			accuracy = naccurate/len(df)
			time_taken = round(time.time() - start,4)
			print("Epoch : " + str(epoch + 1) + " | Accuracy : " + str(accuracy) + " | Time : " + str(time_taken) + " s")
		self.b_values = b_values

	def predict(self,data):
		#urutannya yg padanan b0, b1, dst
		data = [1] + data
		z = dot(data,self.b_values)
		pred = sigmoid(z)
		if pred < 0.5:
			return self.y_vals[0]
		else:
			return self.y_vals[1]