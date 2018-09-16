'''Goals of this receipe :
1. Import and work with dataset.
2. Train classifier.
3. Predict label.
4. Visualize the tree. '''

# 1. Importing the iris datasets
from sklearn.datasets import load_iris
#loaded the dataset
iris = load_iris()

#Knowing the data
print('Feature Names : ', iris.feature_names)
print('Label Names : ', iris.target_names)

print('features of data point 1 : ',iris.data[0])
print('label of data point 1 : ', iris.target[0])

#printing the entire datasets
#for i in range(len(iris.target)) :
#    print( 'flower %d :  label %s :  features %s : ' %( i, iris.target[i], iris.data[i] ) )


#2. Train a classifier (in this example : Decision Tree )
'''Before training the classifier the data must be split. This data should be kept seperated so that it can be used
for testing the classifier later '''

import numpy as np
from sklearn import tree

#for now, only 3 datapoints(one from each class is seperated)
test_ids = [0, 50, 100]

train_data = np.delete(iris.data, test_ids, axis = 0)
train_target = np.delete(iris.target, test_ids)

test_data = iris.data[test_ids]
test_target = iris.target[test_ids]

#Now its time to train the decision tree
clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)
print('The decision tree classifier is trained now.')

#3. Predicting the label for new flower(test data)

print('Target Values : ', test_target)
print('Prediction of the classifier : ', clf.predict(test_data))


#4. Visualizing the decision tree classifier for the given datasets
from sklearn.externals.six import StringIO
import pydotplus

dot_data = StringIO()
tree.export_graphviz(clf,
                         out_file = dot_data,
                         feature_names=iris.feature_names,
                         class_names=iris.target_names,
                         filled=True, rounded=True,
                         impurity = False)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris.pdf")
