from preprocess import transform
from preprocess import fill_missing
import numpy as np
from sklearn import preprocessing
import naive_bayes,lr
from sklearn.linear_model import LogisticRegression as LR
from sklearn.naive_bayes import GaussianNB,MultinomialNB, BernoulliNB
import naive_bayes as nb
import sklearn.ensemble
import sklearn.svm
import sklearn.model_selection as cross
import plot.bar_chart as bar
import plot.histogram as hg
import plot.parallel_coordinates as pc #gender problem
import plot.PCA_biplot as pb
import plot.pie_chart as pie
import plot.scatter_plot as scp
import plot.SVM_visual as sv
import random
from sklearn import metrics
import pandas as pd
import time
import csv

def main():

    # plot part

    filename_train = './data/train.csv'
    train_dataset = transform(filename_train, drop=True)
    X = train_dataset['data']
    y = train_dataset['target']
    pb.plot(filename_train)
    bar.plot(X,y)
    sv.plot(X, y)
    pc.plot(X, y)
    hg.plot(X)
    pie.plot(X,y)
    scp.plot(X)


    # predict part
    # load training data
    filename_train = './data/train.csv'
    filename_test = './data/test.csv'
    train_dataset = transform(filename_train)
    test_dataset = transform(filename_test)
    X = train_dataset['data']
    y = train_dataset['target']
    '''predict'''
    X_predict = test_dataset['data']
    test_id = test_dataset['id']
    print(np.shape(test_id))

    '''open predict.csv'''
    csvfile_lr = open('./predictions/lr_predictions.csv', 'w', newline='')
    csvfile_nb = open('./predictions/nb_predictions.csv', 'w', newline='')
    csvfile_svm = open('./predictions/svm_predictions.csv', 'w', newline='')
    csvfile_rf = open('./predictions/rf_predictions.csv', 'w', newline='')

    writer_lr = csv.writer(csvfile_lr)
    writer_nb = csv.writer(csvfile_nb)
    writer_svm = csv.writer(csvfile_svm)
    writer_rf = csv.writer(csvfile_rf)
    writer_lr.writerow(['UserID', 'Happy'])
    writer_nb.writerow(['UserID', 'Happy'])
    writer_svm.writerow(['UserID', 'Happy'])
    writer_rf.writerow(['UserID', 'Happy'])
    # print('ww',np.shape(test_id)[1])
    m = np.shape(test_id)[0]
    test_id = np.reshape(test_id, (m, 1))

    # fill in missing data (optional)

    X_full = fill_missing(X, 'mean', True)

    # X_full = np.delete(X_full,108,1)
    y = X_full[:, 6]

    # origin data
    X_full = np.delete(X_full, 6, 1)
    X_predict = fill_missing(X_predict, 'mean', True)

    # own_LR ,preprocessing first
    X_predict_lr = preprocessing.scale(X_predict)
    X_full_lr = preprocessing.scale(X_full)
    lr_model = lr.LogisticRegression()
    lr_model.fit(X_full_lr, y, 0.001, 100)
    p = lr_model.predict(X_predict_lr)
    p = np.reshape(p, (np.shape(p)[0], 1))
    m = np.shape(p)[0]
    for i in range(m):
        p_ = p[i]
        id_ = test_id[i]
        writer_lr.writerow([int(id_), int(p_)])
    print('finish lr')

    # naive bayes
    nb_model = naive_bayes.NaiveBayes()
    nb_model.fit(X_full, y)
    p = nb_model.predict(X_predict)
    p = np.reshape(p, (np.shape(p)[0], 1))
    m = np.shape(p)[0]
    for i in range(m):
        p_ = p[i]
        id_ = test_id[i]
        writer_nb.writerow([int(id_), int(p_)])
    print('finish nb')

    #########svm###########
    svm_model = sklearn.svm.SVC(kernel='linear', random_state=0)
    svm_model.fit(X_full, y)
    p = svm_model.predict(X_predict)
    p = np.reshape(p, (np.shape(p)[0], 1))
    m = np.shape(p)[0]
    for i in range(m):
        p_ = p[i]
        id_ = test_id[i]
        writer_svm.writerow([int(id_), int(p_)])
    print('finish svm')
    model = MultinomialNB
    # random forest
    rf_model = sklearn.ensemble.RandomForestClassifier(random_state=0)
    rf_model.fit(X_full, y)
    p = svm_model.predict(X_predict)
    p = np.reshape(p, (np.shape(p)[0], 1))
    m = np.shape(p)[0]
    for i in range(m):
        p_ = p[i]
        id_ = test_id[i]
        writer_rf.writerow([int(id_), int(p_)])
    print('finish rf')

    csvfile_lr.close()
    csvfile_nb.close()
    csvfile_svm.close()
    csvfile_rf.close()





    ## get predictions
    """ your code here """

if __name__ == '__main__':
    main()

'''
 X_train_1, X_test_1 = cross.train_test_split(X_full, random_state=10)
    y_train_1 = X_train_1[:, 6]
    X_train_1 = np.delete(X_train_1, 6, 1)
    y_test_1 = X_test_1[:, 6]
    X_test_1 = np.delete(X_test_1, 6, 1)



       for i in range (1000):
        model = LR(random_state=i)
        model.fit(X_train, y_train)
        pred_1 = model.predict(X_test)
        #print('LR', metrics.accuracy_score(pred_1, y_test))
        accuracy = metrics.accuracy_score(pred_1, y_test)
        if(accuracy > max[0]):
            max[0] = accuracy
            label[0]=i

        rf_model = sklearn.ensemble.RandomForestClassifier(random_state=i)
        rf_model.fit(X_train, y_train)
        y_pred = rf_model.predict(X_test)
        #print('RF', metrics.accuracy_score(y_pred, y_test))
        accuracy = metrics.accuracy_score(pred_1, y_test)
        if (accuracy > max[1]):
            max[1] = accuracy
            label[1] = i
              if (accuracy > max):
        max = accuracy
        label = i

    print (max, label)

'''
'''
label = 0
    max = 0
    X_train, X_test = cross.train_test_split(X_full, random_state=485)  # 485
    y_train = X_train[:, 6]
    X_train = np.delete(X_train, 6, 1)
    y_test = X_test[:, 6]
    X_test = np.delete(X_test, 6, 1)

    model = LR(random_state=999)  # 211
    model.fit(X_train, y_train)
    pred_1 = model.predict(X_test)
    time1 = time.time()
    print('LR accuracy : ', metrics.accuracy_score(pred_1, y_test))

    model = GaussianNB()
    model.fit(X_train, y_train)
    pred_1 = model.predict(X_test)
    print('GNB', metrics.accuracy_score(pred_1, y_test))

    model = MultinomialNB()
    model.fit(X_train, y_train)
    pred_1 = model.predict(X_test)
    print('MNB', metrics.accuracy_score(pred_1, y_test))

    model = BernoulliNB()
    model.fit(X_train, y_train)
    pred_1 = model.predict(X_test)
    print('BNB', metrics.accuracy_score(pred_1, y_test))


    model = sklearn.svm.SVC(kernel='linear',random_state= 233)  # kernel ='polynomial',kernel ='sigmoid',kernel ='rbf',
    model.fit(X_train, y_train)
    pred_1 = model.predict(X_test)
    print('SVC_linear', metrics.accuracy_score(pred_1, y_test))
    time0 = time.time()
    model = sklearn.svm.SVC(kernel='poly',random_state= 233)  # kernel ='polynomial',kernel ='sigmoid',kernel ='rbf',
    model.fit(X_train, y_train)
    pred_1 = model.predict(X_test)
    time1 = time.time()
    #print('time for ploy : ', time1 - time0)
    print('SVC_poly', metrics.accuracy_score(pred_1, y_test))
    model = sklearn.svm.SVC(kernel='rbf')  # kernel ='polynomial',kernel ='sigmoid',kernel ='rbf',
    model.fit(X_train, y_train)
    pred_1 = model.predict(X_test)
    print('SVC_rbf', metrics.accuracy_score(pred_1, y_test))


     time0 = time.time()
    model = LR(random_state=999)  # 211
    model.fit(X_train, y_train)
    pred_1 = model.predict(X_test)
    time1 = time.time()

    print('time for LR : ', time1 - time0)
    print('LR accuracy : ', metrics.accuracy_score(pred_1, y_test))

    time0 = time.time()
    selfmodel = lr.LogisticRegression()
    selfmodel.fit(preprocessing.scale(X_train), y_train, 0.01, 100)
    pred_1 = selfmodel.predict(preprocessing.scale(X_test))
    time1 = time.time()
    print ('time for self LR : ',time1-time0)
    print('self LR accuracy : ', metrics.accuracy_score(pred_1, y_test))

    rf_model = sklearn.ensemble.RandomForestClassifier(random_state=0)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    print('RF', metrics.accuracy_score(y_pred, y_test))

    model = LR(random_state=0)
    model.fit(X_train, y_train)
    pred_1 = model.predict(X_test)
    print('LR', metrics.accuracy_score(pred_1, y_test))

    model = GaussianNB() #211
    model.fit(X_train, y_train)
    pred_1 = model.predict(X_test)
    print('GNB', metrics.accuracy_score(pred_1, y_test))


    model = BernoulliNB() #103
    model.fit(X_train, y_train)
    pred_1 = model.predict(X_test)
    print('BNB', metrics.accuracy_score(pred_1, y_test))

    model = MultinomialNB() #508
    model.fit(X_train, y_train)
    pred_1 = model.predict(X_test)
    print('MNB', metrics.accuracy_score(pred_1, y_test))
    '''
'''
    model = sklearn.svm.SVC(kernel='linear')  # kernel ='polynomial',kernel ='sigmoid',kernel ='rbf',
    model.fit(X_train, y_train)
    pred_1 = model.predict(X_test)
    print('SVC_linear',metrics.accuracy_score(pred_1, y_test))

    model = sklearn.svm.SVC(kernel='poly')  # kernel ='polynomial',kernel ='sigmoid',kernel ='rbf',
    model.fit(X_train, y_train)
    pred_1 = model.predict(X_test)
    print('SVC_poly',metrics.accuracy_score(pred_1, y_test))

    model = sklearn.svm.SVC(kernel='rbf')  # kernel ='polynomial',kernel ='sigmoid',kernel ='rbf',
    model.fit(X_train, y_train)
    pred_1 = model.predict(X_test)
    print('SVC_rbf',metrics.accuracy_score(pred_1, y_test))
    '''
'''
    print('Train the logistic regression classifier')
    model = nb.NaiveBayes()
    model.fit(X_full,y,method = 'gnb')
    y_pred = model.predict(X_full,method='gnb')
    count = 0
    for i in range(0, np.shape(y)[0]):
        if (y_pred[i] == y[i]):
            count = count + 1

    print(count/np.shape(y)[0])
    classifier = MultinomialNB()
    classifier.fit(X_full,y)
    count = 0
    y_pred_1 = classifier.predict(X_full)
    for i in range(0, np.shape(y)[0]):
        if (y_pred_1[i] == y[i]):
            count = count + 1

    print(count/np.shape(y)[0])
    """ your code here """
'''
'''
    lr_model = LR()
    lr_model.fit(X_full,y)
    print (np.shape(lr_model.coef_))

    model = lr.LogisticRegression()
    model.fit(X_full,y)
    y_pred = model.predict(X_full)
    count = 0
    for i in range (0,np.shape(y)[0]):
        if(y_pred[i] == 1):
            count = count +1

    print (count)

    print (np.sqrt(np.sum(np.square(lr_model.coef_ - model.get()))))

    ### use the naive bayes
    print('Train the naive bayes classifier')
    """ your code here """
'''
'''
    classifier = lr.LogisticRegression()  # 0.5647
    classifier.fit(X_full, y)
    y_pred = classifier.predict(X_full)
    print(y_pred)
    count =0
    for i in range(0, np.shape(y_pred)[0]):
        if (y_pred[i] == y[i]):
            count = count + 1
    print(count/np.shape(y)[0])
'''
'''
    #classifier = GaussianNB() #0.6560
    #classifier = naive_bayes.NaiveBayes() #56.427
    # classifier = MultinomialNB()  #0.6692
    #classifier = naive_bayes.NaiveBayes()
    #classifier = BernoulliNB() #0.6657   56.4276048714479% 43.5723951285521%
    classifier = lr.LogisticRegression() #0.5647
    #0.697428
    classifier = sklearn.svm.SVC(kernel='linear') #0.69864
    #classifier = sklearn.ensemble.RandomForestClassifier()
    classifier = lr.LogisticRegression()
    classifier.fit(X_full, y)
    y_pred =  classifier.predict(X_full)
    print (y_pred)

    classifier = naive_bayes.NaiveBayes()
    classifier.fit_gnb(X_full, y)
    y_pred_1 = classifier.predict_gnb(X_full)
    print(y_pred_1)
    count =0

    for i in range (0,np.shape(y_pred)[0]):
        if (y_pred[i] == 1):
            count = count +1
    print (count)

    classifier = naive_bayes.NaiveBayes()
    classifier.fit_mnb(X_full, y)
    y_pred = classifier.predict_mnb(X_full)
    print(y_pred)



    ## use the svm
    print('Train the SVM classifier')
    """ your code here """
    # svm_model = ...

    ## use the random forest
    print('Train the random forest classifier')
    """ your code here """
    # rf_model = ...

    svm_model = sklearn.svm.SVC(kernel='linear')  # kernel ='polynomial',kernel ='sigmoid',kernel ='rbf',

    svm_model.fit(X, y)
    pred_y = svm_model.predict(X)
    print(pred_y)

    rf_model = sklearn.ensemble.RandomForestClassifier()
    rf_model.fit(X, y)
    pred_y = svm_model.predict(X)
    print(pred_y)
'''
