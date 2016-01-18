import os
import re
import wikipedia as wiki
from urllib.request import urlopen
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from math import log
from nltk.stem.snowball import SnowballStemmer


##
##    Word tokenization, steaming, removing stop-words
##
def tokenize(review, ngram, remove_stopwords = True, do_stem = True):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    # 1. Remove non-letters
    if do_stem:
        stemmer = SnowballStemmer("english")
    review_text = re.sub("[^a-zA-Z]"," ", review)
    # 2. Convert words to lower case and split them
    words = review_text.lower().split()
    # 3. Optionally remove stop words (true by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
        if do_stem:
            words = list(map (lambda x: stemmer.stem(x), words))
    # 4. Find n-grams
    words_ngrams = find_ngrams(words, ngram)

    # 5. Return a list of words
    return words_ngrams

def ensure_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


##
##    N-grams search
##
def find_ngrams(input_list, n):
  n_gr_all = []
  for i in range(1,n+1,1):
      n_gr = list(zip(*[input_list[i:] for i in range(i)]))
      n_gr = list(map(lambda x: " ".join (x).strip(),n_gr))
      n_gr_all = n_gr_all + n_gr
    
  return n_gr_all


## Collecting data from URLs
def get_keyword_from_url_topic(url_topic):
    # Topic includes: Earth Science, Life Science, Physical Science, Biology, Chemestry and Physics
    lst_url = []
    html = urlopen(url_topic).read()
    soup = BeautifulSoup(html, 'html.parser')
    for tag_h3 in soup.find_all('h3'):
        url_res =  ' '.join(tag_h3.li.a.get('href').strip('/').split('/')[-1].split('-'))
        lst_url.append(url_res)
    return lst_url


def get_save_wiki_docs(keywords, save_folder = 'data/wiki_data/'):
    
    ensure_dir(save_folder)
    
    n_total = len(keywords)
    for i, kw in enumerate(keywords):
        kw = kw.lower()
        print (i, n_total, i * 1.0 / n_total, kw)
        try:
            content = wiki.page(kw).content.encode('ascii', 'ignore')
        except wiki.exceptions.DisambiguationError as e:
            print ('DisambiguationError', kw)
        except:
            print ('Error', kw)
        if not content:
            continue
        with open(os.path.join(save_folder, '_'.join(kw.split()) + '.txt'), 'wb') as f:
                f.write(content)

 




###
###   Calculate tfidf matrix
###     
def get_docstf_idf(dir_data, n_gram):
    """ indexing wiki pages:
    returns {document1:{word1:tf, word2:tf ...}, ....},
            {word1: idf, word2:idf, ...}"""

    print ("running tf_idf")
    docs_tf = {}
    idf = {}
    vocab = set()
    doc_num = 0
    for fname in os.listdir(dir_data):
        dd = {}
        total_w = 0
        path = os.path.join(dir_data, fname)
        doc_num = doc_num +1
        if (doc_num % 200 ==0):
            print (doc_num)
        #print (fname)
        #for index, line in enumerate(open(path)):
        lst = tokenize(open(path).read(), ngram = n_gram)
        for word in lst:
                vocab.add(word)
                dd.setdefault(word, 0)
                idf.setdefault(word, 0)
                dd[word] += 1
                if dd[word]==1:
                    idf[word] += 1
                total_w += 1 
                
        
        for k, v in dd.items(): 
            dd[k] = (1.* v / total_w)**0.5
        
        docs_tf[fname] = dd
    
    print ("calculating tf-idf: ")
    for w in list(vocab):
        docs_with_w = idf[w]
        # for path, doc_tf in docs_tf.items():
        #    if w in doc_tf:
        #        docs_with_w += 1
        idf[w] = log(len(docs_tf)/(1+docs_with_w))+1


    return docs_tf, idf


def get_docs_importance_for_question(question, dosc_tf, word_idf, n_gram, max_docs = None ):
    question_words = set(tokenize(question, ngram = n_gram))
    #go through each article
    doc_importance = []

    for doc, doc_tf in dosc_tf.items():
        doc_imp = 0
        for w in question_words:
            if w in doc_tf:
                doc_imp += doc_tf[w]  * word_idf[w]
        doc_importance.append((doc, doc_imp))
    
    #sort doc importance    
    doc_importance = sorted(doc_importance, key=lambda x: x[1], reverse = True)
    if max_docs:
        return doc_importance[:max_docs]
    else:
        return doc_importance
