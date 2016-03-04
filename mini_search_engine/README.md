Files:
P1.py (contains the source code)

Methods:
query(qstring): return the document that has the highest similarity score with respect to 'qstring'.
getcount(token): return the total number of occurrences of a token in all documents.
getidf(token): return the inverse document frequency of a token. If the token doesn't exist in the corpus, return 0.
docdocsim(filename1,filename2): return the cosine similarity betwen two speeches (files).
querydocsim(qstring,filename): return the cosine similairty between a query string and a document.

Instructions:
import P1
P1.query("have been in a loss")
similarly execute other queries
