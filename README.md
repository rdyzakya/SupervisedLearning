# Supervised Learning
Repositori ini berisikan model classifier sederhana yakni K-Nearest Neighbor, Logistic Regression, dan ID3 Decision Tree

## How To Use
Berikut adalah tatacara pemakaiannya:
1. Pastikan Python telah tersedia (dibuat dan dites menggunakan Python 3.8.5)
2. Pastikan library pandas dan numpy tersedia
3. Buatlah sebuah file notebook (ipynb), bisa juga dipakai di file .py sesuai kebutuhan
4. Lakukan import source code model

```python
import src.knn as knn
import src.id3 as id3
import src.logreg as lr
```

5. Instansiasi dataframe menggunakan pandas

```python
import pandas as pd
df = pd.read_csv('namafile.csv')
```

### Penggunaan KNN

6.a. Instansiasi model

```python
#kolom ke-0,1,2,3 merupakan kolom fitur, kolom ke-4 merupakan kolom label/target
clf = knn.KNNClassifier(df,x_labels=[0,1,2,3],y_label=4)
```

6.b. Lakukan predict kepada data yang ingin dipredict

```python
#satu data point
data_point = [1.52369, 13.44, 0.0, 1.58]
result = clf.predict(data_point,k=11,print_time=True)
print(result)

#dataframe
result = df.apply(lambda x: clf.predict(x[['A','B','C','D']].to_list(),k=7),axis=1)
print(result)
```

### Penggunaan Logistic Regression

7.a. Instansiasi model

```python
clf = lr.LogisticRegression()
```

7.b. Lakukan training pada model menggunakan dataset

```python
clf.fit(df,x_labels=[0,1],y_label=2,alpha=0.9,epochs=100)
```
```
output:
Epoch : 1 | Accuracy : 0.5 | Time : 0.0069 s
Epoch : 2 | Accuracy : 0.6 | Time : 0.0045 s
Epoch : 3 | Accuracy : 0.8 | Time : 0.0079 s
Epoch : 4 | Accuracy : 0.9 | Time : 0.0081 s
Epoch : 5 | Accuracy : 1.0 | Time : 0.009 s
Epoch : 6 | Accuracy : 1.0 | Time : 0.0083 s
Epoch : 7 | Accuracy : 1.0 | Time : 0.008 s
Epoch : 8 | Accuracy : 1.0 | Time : 0.0053 s
Epoch : 9 | Accuracy : 1.0 | Time : 0.0045 s
Epoch : 10 | Accuracy : 1.0 | Time : 0.0043 s
...
```

7.c. Lakukan predict kepada data yang ingin diklasifikasi

```python
#satu datapoint
clf.predict([-0.136471,0.632003])

#dataframe
df['label'] = df.apply(lambda x: clf.predict(df[['A','B']].to_list()),axis=1)
```

### Penggunaan ID3 Decision Tree

8.a. Instansiasi model id3

```python
clf = id3.DecisionTree()
```

8.b. Training model

```python
clf.fit(df,x_labels=[0,1,2,3],y_label=4)
```

8.c. Output tree yang dibentuk jika ingin melihatnya

```
clf.tree.print()

#contoh output
Outlook
  - sunny
    Humidity
      - high
      (no)
      - normal
      (yes)
  - overcast
  (yes)
  - rain
    Wind
      - weak
      (yes)
      - strong
      (no)
```

8.d. Lakukan prediksi/klasifikasi

```python
#datapoint
clf.predict(['overcast','hot','high','weak'])

#dataframe
df.apply(lambda x : classifier.predict(x[['Outlook', 'Temperature', 'Humidity', 'Wind']].to_list()),axis=1)
```
