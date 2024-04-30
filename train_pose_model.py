import numpy as np
import glob
import cv2
import os
import sys

from sklearn import model_selection
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

import pickle
from sklearn.metrics import confusion_matrix,classification_report

test_data_dir = "./test_data/images/"

feature_set_files = glob.glob(test_data_dir+"*.npy")
pred_class_files = glob.glob(test_data_dir+"*.txt")

input_feature_set = []
pred_classes = []
for feature_file in feature_set_files:

    cur_idx = ((feature_file.split(".")[0]).split("/")[-1]).split("_")[0]
    print(cur_idx)
    pred_file = test_data_dir+str(cur_idx)+"_posture.txt"
    if os.path.isfile(pred_file):

        feature_set_final = []

        feature_set_final = (np.load(feature_file))
        
        row=len(feature_set_final)
        column=len(feature_set_final[0])
        print(f'Final Feature Set - Rows:{row}, Column:{column}')
        feature_set_final = np.asarray(feature_set_final,dtype=np.uint32).flatten()
        with open(pred_file,'r') as f:
                    pred_class = f.readline()
                    f.close()
                    if pred_class=='outside':
                        pred_classes.append(0)
                    elif pred_class=='edge':
                        pred_classes.append(1)
                    elif pred_class=='bed_sleep':
                        pred_classes.append(2)
                    elif pred_class=='bed_sit':
                        pred_classes.append(3)
                    elif pred_class=='chair':
                        pred_classes.append(4)
                    elif pred_class=='fall':
                        pred_classes.append(5)
        if len(feature_set_final) != 44:
             sys.exit(-1)            
        input_feature_set.append(feature_set_final)

lens = [ len(e) for e in input_feature_set]
print(set(lens))
#print("Train pred shape : "+str(np.asarray(input_feature_set,dtype=np.uint32).shape))


# Split  data into train/test data sets , 30% testing and 70% train
X_train, X_test, Y_train, Y_test = model_selection.train_test_split (input_feature_set, 
                                                                     pred_classes, 
                                                                     test_size=0.1,
                                                                     random_state=42)
# Build an RandomForestClassifier model 
clf_rf = RandomForestClassifier(n_estimators=30, max_depth=4, 
                                max_features = 0.4, random_state=42,
                                verbose=100,n_jobs=4)

X_train_np = np.asarray(X_train,dtype=np.uint32)
Y_train_np =(np.asarray(Y_train,dtype=np.uint32)).ravel()
clf_rf.fit(X_train_np,Y_train_np)

#Check accuracy score
clf_rf.score(np.asarray(X_test), np.asarray(Y_test))
print(clf_rf.score(np.asarray(X_test), np.asarray(Y_test)))



# Validate robustness of above model using K-Fold Cross validation technique
scores_res = model_selection.cross_val_score(clf_rf, np.asarray(input_feature_set), 
                                             (np.asarray(pred_classes)).ravel(), 
                                             cv=5, verbose=100,n_jobs=4)

# Print the accuracy of each fold (i.e. 5 as above we asked cv 5)
print(scores_res)

# And the mean accuracy of all 5 folds.
print(scores_res.mean())

filename = 'room_posture_estimtion_OFC.sav'
pickle.dump(clf_rf, open(filename, 'wb'))

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)
print(result)

test_preds = loaded_model.predict(X_test)
confusion_matrix_graph = confusion_matrix(Y_test, test_preds)
classification_report_res = classification_report(Y_test, test_preds)
print('\n*Classification Report:\n', classification_report_res)
print('\n*Confusion Matrix Report:\n', confusion_matrix_graph)