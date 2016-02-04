# -*- coding: utf-8 -*-

try:
    from utils import ipyth_utils_par
except:
    import ipyth_utils_par

from multiprocessing import Pool



def unwrapFunc_similar_score(arg):
    return ck12_predict_cl.similar_score_single(**arg)



class ck12_predict_cl():
    def __init__ (self):
        self.docs_tf = dict()
        self.words_idf = dict()
        self.ngram = -1
        #index docs
        pass
    

    def tf_idf_dict (self, wiki_docs_dir, n_gram, workers):

        self.docs_tf, self.words_idf = ipyth_utils_par.get_docstf_idf_parallel(wiki_docs_dir,n_gram, workers)
        self.max_n_gram = n_gram
        pass
   


    def similar_score_paral(self, data, docs_per_q, n_gram, workers):

        print (data)
        print ("Threading started:")
        arg_pool = list(map(lambda x: {'self':self, 'data':x, 'docs_per_q':docs_per_q, 'n_gram':n_gram}, data))
        print (arg_pool)
        pool = Pool(processes = workers)
        docs_predicitions = pool.map(unwrapFunc_similar_score, arg_pool)
        pool.close()
        pool.join() 
        print ("Threading finished")
        return (docs_predicitions)



    def similar_score_single(self, data, docs_per_q, n_gram):
        res = []
        doc_score = []
        print (data)
        row = data

        #for index, row in data.iterrows():
        #get answers words
        w_A = set(ipyth_utils_par.tokenize(row['answerA'],n_gram))
        w_B = set(ipyth_utils_par.tokenize(row['answerB'],n_gram))
        w_C = set(ipyth_utils_par.tokenize(row['answerC'],n_gram))
        w_D = set(ipyth_utils_par.tokenize(row['answerD'],n_gram))

        sc_A = 0
        sc_B = 0
        sc_C = 0
        sc_D = 0

        q = row['question']

        for d in list(zip(*ipyth_utils_par.get_docs_importance_for_question(q, self.docs_tf, self.words_idf, n_gram = n_gram, max_docs = docs_per_q)))[0]:
            for w in w_A:
                if w in self.docs_tf[d]:
                    sc_A += 1. * self.docs_tf[d][w] * self.words_idf[w]
            for w in w_B:
                if w in self.docs_tf[d]:
                    sc_B += 1. * self.docs_tf[d][w] * self.words_idf[w]
            for w in w_C:
                if w in self.docs_tf[d]:
                    sc_C += 1. * self.docs_tf[d][w] * self.words_idf[w]
            for w in w_D:
                if w in self.docs_tf[d]:
                    sc_D += 1. * self.docs_tf[d][w] * self.words_idf[w]

        res.append(['A','B','C','D'][np.argmax([sc_A, sc_B, sc_C, sc_D])])
        doc_score.append([sc_A, sc_B, sc_C, sc_D])
        return res, doc_score



if __name__ == '__main__':
    import pandas as pd
    import time


    fname_str = 'joined_set.tsv'
    data = pd.read_csv('../../data/' + fname_str, sep = '\t')

    wiki_docs_dir = '../../data/wiki_data'
    N_WORKERS = 1

    print("Building TF-idf model")
    start_time = time.time()
    ck12_prediction = ck12_predict_cl ()
    ck12_prediction.tf_idf_dict(wiki_docs_dir, n_gram = 3, workers=N_WORKERS)
    print ("tf-idf collected")
    print ("elapsed time: ",round((time.time()-start_time)/60,2))


    ## PREDICTING DATA

    ## predict
    print ("run: predicting data")
    start_time = time.time()
    #print (data)
    docText = '\n\n'.join([paragraph.text.encode('utf-8') for paragraph in document.paragraphs])
    print (docText)

    res, prob_scores = ck12_prediction.similar_score_paral(data, docs_per_q = 10, n_gram = 3, workers=N_WORKERS)
    print ("elapsed time: ",round((time.time()-start_time)/60,2))
    print ("finished predicting probabilities")