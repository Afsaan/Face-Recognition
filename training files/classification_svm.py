from random import choice
from numpy import load
from numpy import expand_dims
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from matplotlib import pyplot
from numpy import load
from sklearn.metrics import accuracy_score , precision_score , recall_score , f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from sklearn.externals import joblib 
import pickle

names_array = ['Afsan' , 'Amresh' , 'Amritansh' , 'Ayush' , 'Harish' , 'Keyur' , 'Rahul' ]

# load dataset
data = load('3-friends-faces-embeddings.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Dataset: train=%d, test=%d' % (trainX.shape[1], testX.shape[1]))
# normalize input vectors
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)
# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)
testy = out_encoder.transform(testy)
# fit model
model = SVC(kernel='linear', probability=True , )
model.fit(trainX, trainy)

#saving the svm model
joblib.dump(model , 'svm_face_classification.pkl') 

# with open('svm_classification_1.pkl','wb') as fout:
#     pickle.dump(model,fout)

# predict
yhat_train = model.predict(trainX)
yhat_test = model.predict(testX)
# score
score_train = accuracy_score(trainy, yhat_train)
score_test = accuracy_score(testy, yhat_test)
# summarize
print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))
print('\n')
precision_train = precision_score(trainy, yhat_train , average='micro')
precision_test = precision_score(testy, yhat_test , average='micro')
print('Precision: train=%.3f, test=%.3f' % (precision_train*100, precision_test*100))
print('\n')
recall_train = recall_score(trainy, yhat_train , average='micro')
recall_test = recall_score(testy, yhat_test , average='micro')
print('Recall: train=%.3f, test=%.3f' % (recall_train*100, recall_test*100))
print('\n')
f1_train = f1_score(trainy, yhat_train , average='micro')
f1_test = f1_score(testy, yhat_test , average='micro')
print('F1 score: train=%.3f, test=%.3f' % (f1_train*100, f1_test*100))
print('\n')

# test model on a random example from the test dataset

data = load('3-friends-faces-dataset.npz')
testX_faces = data['arr_2']
selection = choice([i for i in range(testX.shape[0])])
random_face_pixels = testX_faces[selection]
random_face_emb = testX[selection]
random_face_class = testy[selection]
random_face_name = out_encoder.inverse_transform([random_face_class])
# prediction for the face
samples = expand_dims(random_face_emb, axis=0)
yhat_class = model.predict(samples)
yhat_prob = model.predict_proba(samples)
# get name
class_index = yhat_class[0]
class_probability = yhat_prob[0,class_index] * 100
predict_names = out_encoder.inverse_transform(yhat_class)
# print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
print('Predicted output: %s (%.3f)' % (names_array[yhat_class[0]], class_probability))
print('Expected: %s' % random_face_name[0])
# plot for fun
pyplot.imshow(random_face_pixels)
title = '%s (%.3f)' % (predict_names[0], class_probability)
pyplot.title(title)
pyplot.show()