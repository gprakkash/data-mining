from os import listdir
from os.path import join
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from math import sqrt, log10
from copy import deepcopy

# document vector in global scope
doc_vector = None
doc_vector_with_term_freq = None

def tokenize(string):
    '''tokenizes a string into a list of tokens'''
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    return tokenizer.tokenize(string)
    

def remove_stopwords(tokens):
    '''removes english stopwords from the list of tokens'''
    stop_words = stopwords.words('english') # gets a list of stopwords
    filtered_tokens = []
    for token in tokens:
        if token not in stop_words:
            filtered_tokens.append(token)
    return filtered_tokens

def stem(tokens):
    '''stems each token'''
    ps = PorterStemmer()
    for i in range(len(tokens)):
        tokens[i] = ps.stem(tokens[i])

def create_vector(tokens):
    '''create a vector of terms and its term frequency'''
    vector = dict.fromkeys(tokens,0)
    for token in tokens:
        vector[token] += 1
    return vector

def process_docs(corpus_root):
    '''read all the files and then performs
    tokenization, stopword removal, stemming, vector creation'''
    # to store each document vector in a dictionary with key its filename
    doc_vector = {}
    for filename in listdir(corpus_root):
        file = open(join(corpus_root, filename), "r", encoding = 'UTF-8')
        docstr = file.read()
        file.close()
        docstr = docstr.lower()
        tokens = tokenize(docstr)
        tokens = remove_stopwords(tokens)
        stem(tokens)
        vector = create_vector(tokens)
        doc_vector[filename] = vector
    return doc_vector

def get_doc_freq(token):
    '''returns the document frequency i.e., the number of documents
    in the corpus in which the token appears'''
    count = 0
    for vector in doc_vector.values():
        if token in vector:
            count += 1
    return count

def getcount(token):
    '''return the total number of occurrences of a
    token in all documents'''
    count = 0
    for vector in doc_vector_with_term_freq.values():
        if token in vector:
            count += vector[token]
    return count

def getidf(token):
    '''computes the idf value of the token'''
    df = get_doc_freq(token)
    N = len(doc_vector)
    if df != 0:
        return log10(N/df)
    else:
        return 0

def normalize_vector(vector):
    '''divides each element of the vector by the magnitude of the vector'''
    mag = 0
    # compute the magnitude of the vector
    for key in vector:
        mag += vector[key]**2
    mag = sqrt(mag)
    # values after normalization
    for key in vector:
        if mag != 0:
            vector[key] = vector[key]/mag

def update_term_weights_for_docs(doc_vector):
    '''converts the term frequency of each document vector to its
    term weight i.e., tf-idf'''
    for docname in doc_vector:
        for token in doc_vector[docname]:
            tf = doc_vector[docname][token]
            if tf != 0:
                tf = 1 + log10(tf)
            else:
                tf = 1
            idf = getidf(token)
            doc_vector[docname][token] = tf * idf
        normalize_vector(doc_vector[docname])

def process_query(qstring):
    '''process the query string'''
    # convert query to lower case
    qstring = qstring.lower()
    # tokenize the query
    tokens = tokenize(qstring)
    # remove stopwords
    tokens = remove_stopwords(tokens)
    # stem
    stem(tokens)
    # create vector
    query_vector = create_vector(tokens)
    return query_vector

def update_term_weights_for_query(query_vector):
    '''converts the term frequency of each element of the
    vector to its term weight calculated by taking log10'''
    for token in query_vector:
        tf = query_vector[token]
        if tf != 0:
            tf = 1 + log10(tf)
        else:
            tf = 1
        query_vector[token] = tf
    # normalize the vector
    normalize_vector(query_vector)

def get_cosine_sim(vector1, vector2):
    '''returns the cosine similarity between vector1 and vector2'''
    cos_sim = 0
    for token in vector1:
        elt1 = vector1[token]
        elt2 = vector2.get(token, 0)
        cos_sim += elt1 * elt2
    return cos_sim

def querydocsim(query_vector, filename):
    '''return the cosine similairty between a
    query string and a document'''
    # doc_col_vector is a column vector which represents each document
    doc_col_vector = doc_vector[filename]
    return get_cosine_sim(query_vector, doc_col_vector)

def query(qstring):
    '''return the document that has the highest similarity score with respect to "query"'''
    query_vector = process_query(qstring)
    update_term_weights_for_query(query_vector)
    doc_with_max_sim = None
    max_cos_sim = 0
    iteration = 1
    # iterates over each document to find the one with max similarity to the query string
    for filename in doc_vector:
        # executed only for the first time
        if iteration == 1:
            iteration = 0
            cos_sim = querydocsim(query_vector, filename)
            if cos_sim >= max_cos_sim:
                max_cos_sim = cos_sim
                doc_with_max_sim = filename
        # executed from the 2nd iteration to the last
        else:
            cos_sim = querydocsim(query_vector, filename)
            if cos_sim >= max_cos_sim:
                max_cos_sim = cos_sim
                doc_with_max_sim = filename
    return doc_with_max_sim

def docdocsim(doc1, doc2):
    '''return the cosine similarity betwen two speeches (files)'''
    doc1_col_vector = doc_vector[doc1]
    doc2_col_vector = doc_vector[doc2]
    return get_cosine_sim(doc1_col_vector, doc2_col_vector)

# set the path to the directory that contains the documents
corpus_root = './presidential_debates'
# corpus_root = './test_files'
doc_vector = process_docs(corpus_root)
doc_vector_with_term_freq = deepcopy(doc_vector)
update_term_weights_for_docs(doc_vector)
