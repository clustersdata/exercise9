# exercise9
exercise9
Iris
Introduction:
This exercise may seem a little bit strange, but keep doing it.
Step 1. Import the necessary libraries
In [13]:
import pandas as pd
import numpy as np
Step 2. Import the dataset from this address.
Step 3. Assign it to a variable called iris
In [3]:
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = pd.read_csv(url)

iris.head()
Out[3]:
5.1	3.5	1.4	0.2	Iris-setosa
0	4.9	3.0	1.4	0.2	Iris-setosa
1	4.7	3.2	1.3	0.2	Iris-setosa
2	4.6	3.1	1.5	0.2	Iris-setosa
3	5.0	3.6	1.4	0.2	Iris-setosa
4	5.4	3.9	1.7	0.4	Iris-setosa
Step 4. Create columns for the dataset
In [5]:
# 1. sepal_length (in cm)
# 2. sepal_width (in cm)
# 3. petal_length (in cm)
# 4. petal_width (in cm)
# 5. class

iris.columns = ['sepal_length','sepal_width', 'petal_length', 'petal_width', 'class']
iris.head()
Out[5]:
sepal_length	sepal_width	petal_length	petal_width	class
0	4.9	3.0	1.4	0.2	Iris-setosa
1	4.7	3.2	1.3	0.2	Iris-setosa
2	4.6	3.1	1.5	0.2	Iris-setosa
3	5.0	3.6	1.4	0.2	Iris-setosa
4	5.4	3.9	1.7	0.4	Iris-setosa
Step 5. Is there any missing value in the dataframe?
In [11]:
pd.isnull(iris).sum()
# nice no missing value
Out[11]:
sepal_length    0
sepal_width     0
petal_length    0
petal_width     0
class           0
dtype: int64
Step 6. Lets set the values of the rows 10 to 29 of the column 'petal_length' to NaN
In [36]:
iris.iloc[10:30,2:3] = np.nan
iris.head(20)
Out[36]:
sepal_length	sepal_width	petal_length	petal_width	class
0	4.9	3.0	1.4	0.2	Iris-setosa
1	4.7	3.2	1.3	0.2	Iris-setosa
2	4.6	3.1	1.5	0.2	Iris-setosa
3	5.0	3.6	1.4	0.2	Iris-setosa
4	5.4	3.9	1.7	0.4	Iris-setosa
5	4.6	3.4	1.4	0.3	Iris-setosa
6	5.0	3.4	1.5	0.2	Iris-setosa
7	4.4	2.9	1.4	0.2	Iris-setosa
8	4.9	3.1	1.5	0.1	Iris-setosa
9	5.4	3.7	1.5	0.2	Iris-setosa
10	4.8	3.4	NaN	0.2	Iris-setosa
11	4.8	3.0	NaN	0.1	Iris-setosa
12	4.3	3.0	NaN	0.1	Iris-setosa
13	5.8	4.0	NaN	0.2	Iris-setosa
14	5.7	4.4	NaN	0.4	Iris-setosa
15	5.4	3.9	NaN	0.4	Iris-setosa
16	5.1	3.5	NaN	0.3	Iris-setosa
17	5.7	3.8	NaN	0.3	Iris-setosa
18	5.1	3.8	NaN	0.3	Iris-setosa
19	5.4	3.4	NaN	0.2	Iris-setosa
Step 7. Good, now lets substitute the NaN values to 1.0
In [39]:
iris.petal_length.fillna(1, inplace = True)
iris
Out[39]:
sepal_length	sepal_width	petal_length	petal_width	class
0	4.9	3.0	1.4	0.2	Iris-setosa
1	4.7	3.2	1.3	0.2	Iris-setosa
2	4.6	3.1	1.5	0.2	Iris-setosa
3	5.0	3.6	1.4	0.2	Iris-setosa
4	5.4	3.9	1.7	0.4	Iris-setosa
5	4.6	3.4	1.4	0.3	Iris-setosa
6	5.0	3.4	1.5	0.2	Iris-setosa
7	4.4	2.9	1.4	0.2	Iris-setosa
8	4.9	3.1	1.5	0.1	Iris-setosa
9	5.4	3.7	1.5	0.2	Iris-setosa
10	4.8	3.4	1.0	0.2	Iris-setosa
11	4.8	3.0	1.0	0.1	Iris-setosa
12	4.3	3.0	1.0	0.1	Iris-setosa
13	5.8	4.0	1.0	0.2	Iris-setosa
14	5.7	4.4	1.0	0.4	Iris-setosa
15	5.4	3.9	1.0	0.4	Iris-setosa
16	5.1	3.5	1.0	0.3	Iris-setosa
17	5.7	3.8	1.0	0.3	Iris-setosa
18	5.1	3.8	1.0	0.3	Iris-setosa
19	5.4	3.4	1.0	0.2	Iris-setosa
20	5.1	3.7	1.0	0.4	Iris-setosa
21	4.6	3.6	1.0	0.2	Iris-setosa
22	5.1	3.3	1.0	0.5	Iris-setosa
23	4.8	3.4	1.0	0.2	Iris-setosa
24	5.0	3.0	1.0	0.2	Iris-setosa
25	5.0	3.4	1.0	0.4	Iris-setosa
26	5.2	3.5	1.0	0.2	Iris-setosa
27	5.2	3.4	1.0	0.2	Iris-setosa
28	4.7	3.2	1.0	0.2	Iris-setosa
29	4.8	3.1	1.0	0.2	Iris-setosa
...	...	...	...	...	...
119	6.9	3.2	5.7	2.3	Iris-virginica
120	5.6	2.8	4.9	2.0	Iris-virginica
121	7.7	2.8	6.7	2.0	Iris-virginica
122	6.3	2.7	4.9	1.8	Iris-virginica
123	6.7	3.3	5.7	2.1	Iris-virginica
124	7.2	3.2	6.0	1.8	Iris-virginica
125	6.2	2.8	4.8	1.8	Iris-virginica
126	6.1	3.0	4.9	1.8	Iris-virginica
127	6.4	2.8	5.6	2.1	Iris-virginica
128	7.2	3.0	5.8	1.6	Iris-virginica
129	7.4	2.8	6.1	1.9	Iris-virginica
130	7.9	3.8	6.4	2.0	Iris-virginica
131	6.4	2.8	5.6	2.2	Iris-virginica
132	6.3	2.8	5.1	1.5	Iris-virginica
133	6.1	2.6	5.6	1.4	Iris-virginica
134	7.7	3.0	6.1	2.3	Iris-virginica
135	6.3	3.4	5.6	2.4	Iris-virginica
136	6.4	3.1	5.5	1.8	Iris-virginica
137	6.0	3.0	4.8	1.8	Iris-virginica
138	6.9	3.1	5.4	2.1	Iris-virginica
139	6.7	3.1	5.6	2.4	Iris-virginica
140	6.9	3.1	5.1	2.3	Iris-virginica
141	5.8	2.7	5.1	1.9	Iris-virginica
142	6.8	3.2	5.9	2.3	Iris-virginica
143	6.7	3.3	5.7	2.5	Iris-virginica
144	6.7	3.0	5.2	2.3	Iris-virginica
145	6.3	2.5	5.0	1.9	Iris-virginica
146	6.5	3.0	5.2	2.0	Iris-virginica
147	6.2	3.4	5.4	2.3	Iris-virginica
148	5.9	3.0	5.1	1.8	Iris-virginica
149 rows × 5 columns
Step 8. Now let's delete the column class
In [40]:
del iris['class']
iris.head()
Out[40]:
sepal_length	sepal_width	petal_length	petal_width
0	4.9	3.0	1.4	0.2
1	4.7	3.2	1.3	0.2
2	4.6	3.1	1.5	0.2
3	5.0	3.6	1.4	0.2
4	5.4	3.9	1.7	0.4
Step 9. Set the first 3 rows as NaN
In [52]:
iris.iloc[0:3 ,:] = np.nan
iris.head()
Out[52]:
sepal_length	sepal_width	petal_length	petal_width
0	NaN	NaN	NaN	NaN
1	NaN	NaN	NaN	NaN
2	NaN	NaN	NaN	NaN
3	5.0	3.4	1.5	0.2
4	4.4	2.9	1.4	0.2
Step 10. Delete the rows that have NaN
In [53]:
iris = iris.dropna(how='any')
iris.head()
Out[53]:
sepal_length	sepal_width	petal_length	petal_width
3	5.0	3.4	1.5	0.2
4	4.4	2.9	1.4	0.2
5	4.9	3.1	1.5	0.1
6	5.4	3.7	1.5	0.2
7	4.8	3.4	1.0	0.2
Step 11. Reset the index so it begins with 0 again
In [56]:
iris = iris.reset_index(drop = True)
iris.head()
Out[56]:
sepal_length	sepal_width	petal_length	petal_width
0	5.0	3.4	1.5	0.2
1	4.4	2.9	1.4	0.2
2	4.9	3.1	1.5	0.1
3	5.4	3.7	1.5	0.2
4	4.8	3.4	1.0	0.2
BONUS: Create your own question and answer it.
In [ ]:
