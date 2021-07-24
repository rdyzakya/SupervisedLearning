# Supervised Learning
Repositori ini berisikan model classifier sederhana yakni K-Nearest Neighbor, Logistic Regression, dan ID3 Decision Tree

## How To Use
Berikut adalah tatacara pemakaiannya:
1. Pastikan Python telah tersedia (dibuat dan dites menggunakan Python 3.8.5)
2. Pastikan library pandas dan numpy tersedia
3. Buatlah sebuah file notebook (ipynb), bisa juga dipakai di file .py sesuai kebutuhan
4. Lakukan import source code model

```
import src.knn as knn
import src.id3 as id3
import src.logreg as lr
```

### Penggunaan KNN
5.a. Instansiasi dataframe menggunakan pandas

```
import pandas as pd
df = pd.read_csv('namafile.csv')
```

5.b. Instansiasi model dengan positional argument secara berurut: df (data frame), x_labels (list numerik dari kolom fitur x), y_label (int index kolom fitur y)

Contoh:
```
#kolom ke-0,1,2,3 merupakan kolom fitur, kolom ke-4 merupakan kolom label/target
clf = knn.KNNClassifier(df,[0,1,2,3],4)
```

5.c. Lakukan predict kepada data yang ingin dipredict

Contoh:

```
#satu data point
data_point = [1.52369, 13.44, 0.0, 1.58]
result = clf.predict(data_point,k=11,print_time=True)
print(result)

#series
result = df.apply(lambda x: clf.predict(x[['A','B','C','D']].to_list(),k=7),axis=1)
print(result)
```
