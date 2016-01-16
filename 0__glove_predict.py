import numpy as np
import pandas as pd
from scipy import linalg
from nltk.corpus import stopwords
import argparse
from utils import tokenize
import codecs
import time

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


def predict_answers(data, word2vec, N):

    stop = stopwords.words('english')

    pred_answs = []
    pred_probs = [["A", "B", "C", "D"]]
    for i in range(data.shape[0]):
        #calculate word2vec for question
        q_vec = np.zeros(N, dtype=float)
        for w in tokenize(data['question'][i]):
            if w.lower() in word2vec and w.lower() not in stop:
                w2 = getword2vecval (N,w.lower(),word2vec)
                q_vec = np.add(q_vec, w2)
        q_vec = q_vec / linalg.norm(q_vec)
    
        #calculate word2vec for answers
        A_vec = np.zeros(N, dtype=float)
        B_vec = np.zeros(N, dtype=float)
        C_vec = np.zeros(N, dtype=float)
        D_vec = np.zeros(N, dtype=float)
        for w in tokenize(data['answerA'][i]):
            if w.lower() in word2vec  and w.lower() not in stop:
                w2 = getword2vecval (N,w.lower(),word2vec)
                #print (w2[0:4])
                A_vec = np.add(A_vec,w2)
    
        for w in tokenize(data['answerB'][i]):
            if w.lower() in word2vec and w.lower() not in stop:
                w2 = getword2vecval (N,w.lower(),word2vec)
                #print (w2[0:4])
                B_vec = np.add(B_vec,w2)
            
        for w in tokenize(data['answerC'][i]):
            if w.lower() in word2vec and w.lower() not in stop:
                w2 = getword2vecval (N,w.lower(),word2vec)
                #print (w2[0:4])
                C_vec = np.add(C_vec,w2)

    
        for w in tokenize(data['answerD'][i]):
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

if __name__ == '__main__':
    #parsing input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--fname', type=str, default='joined_set.tsv', help='file name with data')
    parser.add_argument('--N', type=int, default= 300, help='embeding size (50, 100, 200, 300 only)')
    args = parser.parse_args()
    
    #read data
    data = pd.read_csv('data/' + args.fname, sep = '\t' )
    
    #read glove
    args.N = 300
    word2vec = {}
    glove_data_filename = "glove.6B." + str(args.N) + "d.txt"
    #glove_data_filename = "glove.840B.300d.txt"
    #glove_data_filename = "glove.twitter.27B.200d.txt"
    

    words_reading_err = 0
    with codecs.open("data/glove/" + glove_data_filename ,"r","utf-8") as f:
    #with codecs.open("data/glove/glove.6B.50d.txt","r","utf-8") as f:
        print ("--- reading data ---")
        st_time = time.time()
        for line in f.readlines():
           # print ("sdf")
            l = line.split()
            try:
                word2vec[l[0]] = list(map(float, l[1:]))
            except:
                words_reading_err = words_reading_err + 1


        print ("data loaded")
        print (round((time.time() - st_time)/60,2)," min")
        print ("words reading errors:", words_reading_err)

    #predict
    pred_answs, pred_probs = predict_answers(data, word2vec, args.N)
    print(pred_probs[0:5])
    pred_probs = np.array(pred_probs).flatten()
    pred_probs = np.resize (pred_probs,(len(pred_probs)/4,4))
    print(pred_probs[0:5,:])

    #save prediction
    output_file = 'predictions/prediction_glove_dict_' +glove_data_filename[:-4]  + '.csv'
    output_file_probs = 'predictions/prob_prediction_glove_dict_' +glove_data_filename[:-4]  + '.csv'
    print ("Writing output to: ", output_file)
    pd.DataFrame({'id': list(data['id']),'correctAnswer': pred_answs})[['id', 'correctAnswer']].to_csv(output_file, index = False)
    pd.DataFrame({'id': list(data['id']),'probA': pred_probs[1:,0],'probB': pred_probs[1:,1],'probC': pred_probs[1:,2],'probD': pred_probs[1:,3]}).to_csv(output_file_probs, index = False)