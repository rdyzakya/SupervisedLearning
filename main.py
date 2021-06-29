import src.dataframe as df
import src.id3 as id3

a =df.DataFrame('adult.csv')
b = id3.DecisionTree(a)
b.train([1,3,5,6,7,8,9,13],14)