#!/usr/bin/env python

import sys
import csv
import numpy as np
import math
import matplotlib.pyplot as plt
import pprint
import pickle
import argparse

DEGREES = {'WEAK'   : 0.33,
           'AVERAGE': 0.66,
           'STRONG' : 1.0}

ACTIONS = {'WINCE'  : [0,0,0],
            'SMILE' : [0.5,0,0] ,
            'FROWN' : [0,0.5,0],
            'LAUGH' : [0,0,0.5],
            'GLARE' : [0.5,0.5,0],
            'NOD'   : [0.5,0,0.5],
            'SHAKE' : [0,0.5,0.5],
            'REQUEST FOR BOARD': [0.5,0.5,0.5],
            'EYE-ROLL':[1,0,0],
            'JOY'   :  [0,1,0],
            'SUPRISE': [0,0,1],
            'FEAR'  :  [1,1,0],
            'ANGER' :  [0,1,1],
            'DISGUST': [1,0,1],
            'SADNESS': [0.5,0,0]}
ACT_LIST = ['WINCE', 'NOD', 'SHAKE', 'JOY', "FEAR", "SUPRISE", "ANGER", "DISGUST", "SADNESS"]
WINDOW_DUR = 0.25

def extract_data(files):
    data = []
    for data_file in files.split():
        with open(data_file, 'rb') as f:
            reader = csv.reader(f)
            for row in reader:
                data.append(row)
    return data

def process(files, SVM_DATA_FILE, WINDOW_DUR, MAG_THRESH, plot):
    data = extract_data(files)
    window = []
    o_type_cnt={}.fromkeys(ACT_LIST,0)
    f_type_cnt={}.fromkeys(ACT_LIST,0)
    legend_labels = []
    svm_label = []
    svm_data = []

    for dat in data:
        #dat[0]=Degree 
        o_type_cnt[dat[1]] += 1#dat[1] = Action
        dat[2]=float(dat[2]) #Timestamp (float seconds)
        dat[3]=float(dat[3]) #X
        dat[4]=float(dat[4]) #Y
        dat.append((dat[3]**2. + dat[4]**2.)**(1./2)) #dat[5] = Magnitude
        dat.append(math.atan2(dat[4], dat[3])) #dat[6] = Direction
        color = tuple(ACTIONS[dat[1]]+[DEGREES[dat[0]]])
        if (dat[5]<MAG_THRESH):#Initial filtering
            continue
        
        f_type_cnt[dat[1]] += 1
        window.append(dat)
        while (window[-1][2] - window[0][2]) > WINDOW_DUR:
            window.pop(0)
        dat.append(len(window))#dat[7] = Number of points in window
        movement = [0.,0.]
        for datum in window:
            movement[0] += datum[3]
            movement[1] += datum[4]
        dat.append((movement[0]**2+movement[1]**2)**(1./2))#dat[8] = Mag of window movement
        dat.append(dat[-1]/dat[-2])#dat[9] = Avg Window Movement
        dat.append(math.atan2(movement[1],movement[0]))#dat[10] = Dir of window movement
        if plot: 
            if dat[1] not in legend_labels:
                legend_labels.append(dat[1])
                #plt.polar(dat[-1], dat[-2], '.', color=color, label=dat[1])
                plt.plot(dat[-1], dat[-2], '.', color=color, label=dat[1])
                #plt.plot(dat[-3], ACT_LIST.index(dat[1]), '.', color=color, label=dat[1])
            else:
                #plt.polar(dat[-1], dat[-2], '.', color=color)
                plt.plot(dat[-1], dat[-2], '.', color=color)
                #plt.plot(dat[-3], ACT_LIST.index(dat[1]), '.', color=color)

        if SVM_DATA_FILE is not None:
            #OUTPUT Data in format for Sci-kit Learn
            if dat[1] == 'WINCE':
                svm_label.append(1)
            else:
                svm_label.append(0)
            svm_data.append([dat[5],dat[6],dat[7],dat[9],dat[10]])
    

    print " \r\n"*5
    print "Total Datapoints: ", len(data)
    print " \r\n"
    print "Impact of Filtering:"
    for type in o_type_cnt.keys():
        print "%s: \r\n  %s (%2.2f%%) --> \r\n  %s (%2.2f%%)" %(type, 
                                    o_type_cnt[type], 
                                    (100.*o_type_cnt[type])/len(data),
                                    f_type_cnt[type], 
                                    (100.*f_type_cnt[type])/len(data))
    print " \r\n"*2

    if plot:
        plt.legend(loc=2,bbox_to_anchor=(1,1))

    if SVM_DATA_FILE is not None:      
        svm_output = {'labels':svm_label,
                      'data':svm_data}
        with open('../data/'+SVM_DATA_FILE+'.pkl','wb+') as f_pkl:
            pickle.dump(svm_output, f_pkl)


def create_ROC(filename):
        from scipy import interp

        from sklearn import preprocessing as pps, svm
        from sklearn.metrics import roc_curve, auc
        from sklearn.cross_validation import StratifiedKFold, LeaveOneOut

        with open('../data/svm_data.pkl', 'rb') as f:
            svm_data = pickle.load(f)
        labels = svm_data['labels']
        data = svm_data['data']

        scaler = pps.Scaler().fit(data)
        print "Mean: ", scaler.mean_
        print "Std: ", scaler.std_
        data_scaled = scaler.transform(data)

        classifier = svm.SVC(probability=True)
        classifier.fit(data_scaled, labels)

#print "Support Vectors: \r\n", classifier.support_vectors_
        print "SV's per class: \r\n", classifier.n_support_


###############################################################################
## Code below modified from http://scikit-learn.org/stable/auto_examples/plot_roc_crossval.html#example-plot-roc-crossval-py
        X, y = data_scaled, np.array(labels)
        n_samples, n_features = X.shape
        print n_samples, n_features

###############################################################################
# Classification and ROC analysis
# Run classifier with crossvalidation and plot ROC curves
        cv = StratifiedKFold(y, k=9)

        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, n_samples)
        all_tpr = []
        plt.figure(2)
        for i, (train, test) in enumerate(cv):
            probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
            # Compute ROC curve and area the curve
            fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
            mean_tpr += interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, '--', lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

        plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

        mean_tpr /= len(cv)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        plt.plot(mean_fpr, mean_tpr, 'k-', lw=3,
                label='Mean ROC (area = %0.2f)' % mean_auc)

        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()
        print "Finished!"
        
if __name__=='__main__':
    parser = argparse.ArgumentParser(
                description="Process raw wouse training data to output plots,"
                            "statistics, and SVM-ready formatted data",
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('filename', 
                        help="One or more training data files to process")
    parser.add_argument('-o','--output', dest="SVM_DATA_FILE",
                        help="Output file for SVM-formatted training data") 
    parser.add_argument('-w','--window', default=0.250, type=float,
                        help="Length of time window in seconds")
    parser.add_argument('-t','--threshold', default=2.5, type=float,
                        help="Minimum activity threshold")
    parser.add_argument('-p','--plot', action='store_true',
                        help="Produce plots regarding the data")
    parser.add_argument('-r','--ROC', action='store_true',
                        help="Produce ROC Curve using stratified k-fold crossvalidation")
    args = parser.parse_args()

    print "Parsing data from the following files: \r\n ", args.filename

    process(args.filename, args.SVM_DATA_FILE, args.window, args.threshold, args.plot)
    if args.SVM_DATA_FILE is not None and args.ROC:
        create_ROC(args.SVM_DATA_FILE)