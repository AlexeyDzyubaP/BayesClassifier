from __future__ import division
#from ngram import nGram
from collections import Counter
import os, re
import math as calc
import numpy
from sklearn import metrics
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
#from sklearn.metrics import confusion_matrix as sk_confusion_matrix
#from sklearn.metrics import roc_curve, auc
#import matplotlib.pyplot as plt

class NaiveBayesClassifier():
    """A program to use machine learning techniques with ngram maximum likelihood probabilistic language models as feature to build a bayesian text classifier.
Usage:
#>>> tc = NaiveBayesClassifier('documents/')

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_rt_lm, tpr_rt_lm, label='RT + LR')
plt.plot(fpr_rf, tpr_rf, label='RF')
plt.plot(fpr_rf_lm, tpr_rf_lm, label='RF + LR')
plt.plot(fpr_grd, tpr_grd, label='GBT')
plt.plot(fpr_grd_lm, tpr_grd_lm, label='GBT + LR')

    """
    def __init__(self, location):
        """Constructor method to load training data, and train classifier."""
        #self.ng = nGram(False, False)
        self.labels = ['spmsg', 'legit']
        self.nDirs = 10  # data dirs number
        self.N = 2
        self.fileslistSPM, self.fileslistLGT = self.getDocuments(location)
        acc_All = 0
        plt.figure(1)
        acc_ar = [0]*11
        lbl_ar = [0]*11

        #for i in range(self.nDirs):
        for j in range(11):
            self.words = self.loadDocuments(location, 1, self.fileslistSPM, self.fileslistLGT)
            self.train(1)
            print("Lambda = " + str(1 - j*0.01))
            acc, gr_tr_ar, pred_ar = self.myclassify(location, 1,  1 - j*0.01)

            fpr_rf, tpr_rf, _ = roc_curve(gr_tr_ar, pred_ar)

            lbl = "lambda = " + str(1 - j*0.01)
            #plt.plot(fpr_rf, tpr_rf, label=lbl)
            acc_All += acc
            acc_ar[j] = acc
            lbl_ar[j] = 1 - j*0.01
            print('acc = ', acc)

        plt.plot(lbl_ar, acc_ar)
        #plt.show()
        acc_All /= self.nDirs
        plt.xlabel('Lambda')
        plt.ylabel('Accuracy')
        plt.title('Lambda estimation')
        plt.legend(loc='best')
        plt.show()
        print('acc_All = ', acc_All)
        return

    def train(self, i):
        """Method to train classifier by calculating Prior and Likelihood."""
        self.prior = self.calculatePrior(i)
        #self.unigram = self.createUnigram()
        self.unigram = self.createMyNgram()
        
    def myclassify(self, loc, i_test, lmbd):
        """Method to load test data and classify using training data."""
        
        #print('i_test = ', i_test)
        #print('self.fileslistSPM[i_test] = ', self.fileslistSPM[i_test][3])
        n = self.N
        s_All = 0
        s_Rht = 0
        num = -1
        predictions = [0] * 109
        gr_truth = [0] * 109
        for file in self.fileslistSPM[i_test]:
            num = num+1
            handle = open(loc+'part'+str(i_test+1)+'/'+file, 'r')
            words = handle.read()

            s = words
            #s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s)
            tokens = [token for token in s.split(" ") if token != ""]
            ngrams = zip(*[tokens[i:] for i in range(n)])
            ngrams = [" ".join(ngram) for ngram in ngrams]

            #words = words.split()
            P = dict.fromkeys(self.labels, 0)
            for label in self.labels:
                for word in ngrams:
                    P[label] = P[label] + self.calculateLikelihood(word, label)
                P[label] = P[label] + calc.log(self.prior[label])
            estim = sorted(P, key=P.get, reverse=True)[0]

            gr_truth[num] = 0
            s_All += 1
            if estim == 'spmsg':
                s_Rht += 1
                predictions[num] = 0
            else:
                predictions[num] = 1
            #print('filename = ', file, 'estim = ', estim)
            
        #print('s_Rht = ', s_Rht)


        for file in self.fileslistLGT[i_test]:
            handle = open(loc+'part'+str(i_test+1)+'/'+file, 'r')
            words = handle.read()
            num = num+1
            s = words
            #s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s)
            tokens = [token for token in s.split(" ") if token != ""]
            ngrams = zip(*[tokens[i:] for i in range(n)])
            ngrams = [" ".join(ngram) for ngram in ngrams]
            
            #words = words.split()
            P = dict.fromkeys(self.labels, 0)
            for label in self.labels:
                for word in ngrams:
                    P[label] = P[label] + self.calculateLikelihood(word, label)
                P[label] = P[label] + calc.log(self.prior[label])
            P['legit'] = P['legit']*lmbd
            estim = sorted(P, key=P.get, reverse=True)[0]
            s_All += 1

            gr_truth[num] = 1
            if estim == 'legit':
                s_Rht += 1
                predictions[num] = 1
            else:
                predictions[num] = 0
                print('legit went to spam')
            #print('filename = ', file, 'estim = ', estim)
            
        #print('s_Rht = ', s_Rht)
        return s_Rht/s_All, gr_truth, predictions


    def getDocuments(self, location):
        """Method to retrieve test data."""

        fileslistSPM = [[] for x in range(self.nDirs)]
        fileslistLGT = [[] for x in range(self.nDirs)]
        for i in range(self.nDirs):
            for file in os.listdir(location+'part'+str(i+1)+'/'):
                if 'spmsg' in file:
                    fileslistSPM[i].append(file)
                if 'legit' in file:
                    fileslistLGT[i].append(file)

        #print(fileslistSPM[1])
        return fileslistSPM, fileslistLGT

    def calculatePrior(self, i_test):
        """Method to calculate Prior."""
        prior = dict()
        s_SPM = 0
        s_LGT = 0
        for i in range(self.nDirs):
            if i != i_test:
                s_SPM = s_SPM + len(self.fileslistSPM[i])
                s_LGT = s_LGT + len(self.fileslistLGT[i])
  
        prior['spmsg'] = s_SPM/(s_SPM+s_LGT)
        prior['legit'] = s_LGT/(s_SPM+s_LGT)
        return prior

    def calculateLikelihood(self, word, label):
        """Method to calculate Likelihood."""
        return self.unigramProbability(word, label)

    def unigramProbability(self, word, label):
        """Method to calculate Unigram Maximum Likelihood Probability with Laplace Add-1 Smoothing."""
        return calc.log((self.unigram[label][word]+0.1)/(len(self.words[label])+len(self.unigram[label])))

 
    def createMyNgram(self):
        """Method to create Unigram for each class/label."""
        unigram = dict.fromkeys(self.labels, dict())
        n = self.N
        
        for label in self.labels:
            s = self.words[label]  #.lower()
            #s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s)
            tokens = [token for token in s.split(" ") if token != ""]
            ngrams = zip(*[tokens[i:] for i in range(n)])
            ngrams = [" ".join(ngram) for ngram in ngrams]
            #print(ngrams)
            unigram[label] = Counter(ngrams)
            #print(unigram[label])
        return unigram
           
    def loadDocuments(self, loc, i_test, fileslistSPM, fileslistLGT):
        """Method to load labeled data from the training set."""
        
        wordsTrain = dict.fromkeys(self.labels,"")
        for i in range(self.nDirs):
            if i != i_test:
                for file in fileslistSPM[i]:
                    handle = open(loc+'part'+str(i+1)+'/'+file, 'r')
                    wordsTrain['spmsg'] = wordsTrain['spmsg'] + ' ' + handle.read()
                    handle.close()
                for file in fileslistLGT[i]:
                    handle = open(loc+'part'+str(i+1)+'/'+file, 'r')
                    wordsTrain['legit'] = wordsTrain['legit'] + ' ' + handle.read()
                    handle.close()
                    
        #print(wordsTrain['spmsg'])
        return wordsTrain

#########################################        
tc = NaiveBayesClassifier('C:/Users/Alex/PycharmProjects/BayesITMO2/messages/')
#tc.train()
#tc.classify('documents/comedy/shakespeare-comedy-merchant.txt')
#tc.classify('messages/part1/121spmsg62.txt')
