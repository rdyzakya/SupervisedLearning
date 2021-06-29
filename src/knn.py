#KNN
import math
from statistics import mode

class ColumnNumberError(Exception):
	pass

class KNNClassifier:
	def __init__(self,df,category_column):
		self.df = df
		self.category_column = category_column
		self.categorical_columns = []
		start = 0 if df.columns == None else 1
		for i in range(start,len(df)):
			for j in range(len(df[i])):
				try:
					val = float(df[i][j])
					df[i][j] = val
				except:
					if j not in self.categorical_columns:
						self.categorical_columns.append(j)

	def euclidean_distance(self,data,other):
		deltas = []
		for i in range(len(data)):
			if i not in self.categorical_columns and i != self.category_column:
				deltas.append(data[i]-other[i])
		total_squared = 0
		for j in deltas:
			total_squared += j**2
		return math.sqrt(total_squared)

	def predict(self,data,k):
		if len(data) != len(self.df[0])-1:
			raise ColumnNumberError()

		sorted_df = self.df[1:] if self.df.columns != None else self.df[:]
		for i in range(len(sorted_df)):
			nearest = sorted_df[i]
			for j in range(i+1,len(sorted_df)):
				if (self.euclidean_distance(data,sorted_df[i]) > self.euclidean_distance(data,sorted_df[j])):
					nearest = sorted_df[j]
			sorted_df[j] = sorted_df[i]
			sorted_df[i] = nearest

		categories = []
		offset = 1 if self.df.columns != None else 0
		for num in range(offset,k+offset):
			categories.append(sorted_df[num][self.category_column])
		return mode(categories)