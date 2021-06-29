#dataframe
import csv

class DataFrame:
	def __init__(self,dataset_file,ignore_header=True):
		file = open(dataset_file)
		reader = csv.reader(file, delimiter = ',')
		self.dataset = []
		for row in reader:
			self.dataset.append(row)
		self.columns = self.dataset[0] if ignore_header else None

	def __getitem__(self,key):
		return self.dataset[key]

	def __str__(self):
		a = ""
		for i in range(len(self.dataset)):
			a += str(self.dataset[i])
			if i != len(self.dataset)-1:
				a += "\n"
		return a

	def __len__(self):
		return len(self.dataset)