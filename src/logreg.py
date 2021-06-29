#logistic
import math

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
	def __init__(self,df):
		self.df = df
		for i in range(len(self.df)):
			for j in range(len(self.df[i])):
				try:
					val = float(self.df[i][j])
					self.df[i][j] = val
				except:
					pass
		self.b_values = None
		self.y_vals= None

	#x_columns bentuknya list of index column yg jadi x, y cmn int
	def train(self,x_columns, y_column, alpha, epochs):
		offset = 1 if self.df.columns != None else 0
		self.y_vals = []
		for i in range(offset,len(self.df)):
			val = self.df[i][y_column]
			if val not in self.y_vals:
				self.y_vals.append(val)
		if len(self.y_vals) != 2:
			raise NotBinaryClassificationProblemError()
		else:
			if 1 not in self.y_vals or 0 not in self.y_vals:
				for irow in range(offset,len(self.df)):
					self.df[irow][y_column] = 0 if self.df[irow][y_column] == self.y_vals[0] else 1

		b_values = [0 for i in range(len(x_columns) + 1)]

		for epoch in range(epochs):
			for irow in range(offset,len(self.df)):
				row_vector = [1] + [self.df[irow][j] for j in x_columns]
				z = dot(row_vector,b_values)
				prediction = sigmoid(z)
				for ib in range(len(b_values)):
					b_values[ib] += alpha*(self.df[irow][y_column]-prediction)*prediction*(1-prediction)*row_vector[ib]

			naccurate = 0
			for irow in range(offset, len(self.df)):
				row_vector = [1] + [self.df[irow][j] for j in x_columns]
				z = dot(row_vector,b_values)
				prediction = sigmoid(z)
				if prediction < 0.5 and self.df[irow][y_column] == 0:
					naccurate += 1

				if prediction >= 0.5 and self.df[irow][y_column] == 1:
					naccurate += 1
			accuracy = naccurate/(len(self.df) - offset)
			print("Epoch : " + str(epoch + 1) + " | Accuracy : " + str(accuracy))
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