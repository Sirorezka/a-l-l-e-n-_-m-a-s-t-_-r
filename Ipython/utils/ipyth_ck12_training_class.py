from utils import ipyth_utils_par

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
    
    def similar_score(self, data, docs_per_q, n_gram):
        res = []
        doc_score = [["A","B","C","D"]]
        for index, row in data.iterrows():
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