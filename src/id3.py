#ID3
import math
import pandas as pd
import numpy as np

def log(basis,argument):
	numerator = math.log(argument)
	denominator = math.log(basis)
	return numerator/denominator

def entropy(positive,negative):
	total = positive + negative
	p_pos = positive/total
	p_neg = negative/total
	return (-p_pos)*log(2,p_pos) - p_neg*log(2,p_neg)	

class Tree:
	def __init__(self, majority=None):
		self.leafcolname = None
		self.majority = majority
		self.leaves = {}

	def add_leaf(self,value,leaf):
		self.leaves[value] = leaf

	def leaf(self,value):
		return self.leaves[value]

	def get_leafcolname(self):
		return self.leafcolname

	def set_leafcolname(self,leafcolname):
		self.leafcolname = leafcolname

class NotBinaryClassificationError(Exception):
	pass

class DecisionTree:
	def __init__(self,df):
		self.df = df
		self.y_val = None
		self.tree = Tree()

	def train(self,x_labels,y_label):
		columns = self.df.columns
		x_labels = [columns[i] for i in x_labels]
		y_label = columns[y_label]
		self.y_val = self.df[y_label].unique()
		if len(self.y_val) != 2:
			raise NotBinaryClassificationError()
		processed_data = df[x_labels + y_label]
		recursion(self.tree,processed_data,y_label)

	def recursion(self,tree,df,y_label):
		target_feature = df[y_label].unique()
		if len(target_feature) == 1:
			tree.set_leafcolname(y_label)
			leaf = Tree(target_feature[0])
			tree.add_leaf(target_feature[0],leaf)
			return
		elif len(df.columns) == 1:
			tree.set_leafcolname(y_label)
			leaf = Tree(tree.majority)
			tree.add_leaf(tree.majority,leaf)
			return
		#else

		npos_all = len(df[df[y_label] == self.y_val[1]])
		nneg_all = len(df[df[y_label] == self.y_val[0]])
		entropy_all = entropy(npos_all,nneg_all)
		columns = df.columns
		columns.remove(y_label)
		chosen_col = None
		maximum_gain = -np.inf
		for cols in columns:
			current_column_unique = df[cols].unique()
			average_information = 0
			v_df_m = []
			for u in current_column_unique:
				df_u = df[df[cols] == u]
				del df_u[cols]
				npos_u = len(df_u[df_u[y_label] == self.y_val[1]])
				nneg_u = len(df_u[df_u[y_label] == self.y_val[0]])
				if npos_u >= nneg_u:
					v_df_m.append((u,df_u,self.y_val[1]))
				else:
					v_df_m.append((u,df_u,self.y_val[1]))
				numerator = npos_u + nneg_u
				denominator = npos_all + nneg_all
				average_information += (numerator/denominator) * entropy(npos_u,nneg_u)
			gain = entropy_all - average_information
			if gain > maximum_gain:
				chosen_col = (cols,v_df_m)
				maximum_gain = gain
		tree.set_leafcolname(chosen_col[0])
		for i in chosen_col[1]:
			df_i = i[1]
			val_i = i[0]
			tree_i = Tree(i[2])
			tree.add_leaf(val_i,tree_i)
			recursion(tree.leaf(val_i), df_i, y_label)