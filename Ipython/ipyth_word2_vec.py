import numpy as np
import pandas as pd
from scipy import linalg
from nltk.corpus import stopwords
import argparse
from ipyth_utils import tokenize
import codecs
import time


## return word2vec(word)
def getword2vecval (N,w,word2vec):
    i = 0
    resh_w2v = []
    while resh_w2v ==[] and  i<8:
        resh_w2v = list(word2vec[w.lower()])
        i = i+1

    #print (resh_w2v)
    if resh_w2v == []:
        resh_w2v = np.zeros(N, dtype=float)
        print (w)
    resh_w2v = np.array(resh_w2v)
    return resh_w2v


def evaluate_score (y_model, y_real):
    model_score = sum(y_model==y_real)/len(y_real)
    return model_score
    

##
## N - dimensionality of word2vec 
##
def predict_cosine_answers(data, word2vec, N, ngram):

    stop = stopwords.words('english')

    pred_answs = []
    pred_probs = [["A", "B", "C", "D"]]
    for i in range(data.shape[0]):
        #calculate word2vec for question
        q_vec = np.zeros(N, dtype=float)
        for w in tokenize(data['question'][i], ngram):
            if w.lower() in word2vec and w.lower() not in stop:
                w2 = getword2vecval (N,w.lower(),word2vec)
                q_vec = np.add(q_vec, w2)
        q_vec = q_vec / linalg.norm(q_vec)
    
        #calculate word2vec for answers
        A_vec = np.zeros(N, dtype=float)
        B_vec = np.zeros(N, dtype=float)
        C_vec = np.zeros(N, dtype=float)
        D_vec = np.zeros(N, dtype=float)
        for w in tokenize(data['answerA'][i], ngram):
            if w.lower() in word2vec  and w.lower() not in stop:
                w2 = getword2vecval (N,w.lower(),word2vec)
                #print (w2[0:4])
                A_vec = np.add(A_vec,w2)
    
        for w in tokenize(data['answerB'][i], ngram):
            if w.lower() in word2vec and w.lower() not in stop:
                w2 = getword2vecval (N,w.lower(),word2vec)
                #print (w2[0:4])
                B_vec = np.add(B_vec,w2)
            
        for w in tokenize(data['answerC'][i], ngram):
            if w.lower() in word2vec and w.lower() not in stop:
                w2 = getword2vecval (N,w.lower(),word2vec)
                #print (w2[0:4])
                C_vec = np.add(C_vec,w2)

    
        for w in tokenize(data['answerD'][i], ngram):
            if w.lower() in word2vec and w.lower() not in stop:
                w2 = getword2vecval (N,w.lower(),word2vec)
                #print (w2[0:4])
                D_vec = np.add(D_vec,w2)
    
        A_vec = A_vec / linalg.norm(A_vec) 
        B_vec = B_vec / linalg.norm(B_vec)
        C_vec = C_vec / linalg.norm(C_vec)
        D_vec = D_vec / linalg.norm(D_vec)
        
        #choose question based on cosine distance
        idx = np.concatenate((A_vec, B_vec, C_vec, D_vec)).reshape(4, N).dot(q_vec).argmax()
        probs = np.concatenate((A_vec, B_vec, C_vec, D_vec)).reshape(4, N).dot(q_vec)
        pred_answs.append(["A", "B", "C", "D"][idx])
        pred_probs.append(probs)
        
    return pred_answs, pred_probs
