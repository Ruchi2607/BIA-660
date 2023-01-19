#!/usr/bin/env python
# coding: utf-8

# # HW 4: Natural Language Processing

# <div class="alert alert-block alert-warning">Each assignment needs to be completed independently. Never ever copy others' work or let someone copy your solution (even with minor modification, e.g. changing variable names). Anti-Plagiarism software will be used to check all submissions. No last minute extension of due date. Be sure to start working on it ASAP! </div>

# ## Q1: Extract data using regular expression
# Suppose you have scraped the text shown below from an online source (https://www.google.com/finance/). 
# Define a `extract` function which:
# - takes a piece of text (in the format of shown below) as an input
# - uses regular expression to transform the text into a DataFrame with columns: 'Ticker','Name','Article','Media','Time','Price',and 'Change' 
# - returns the DataFrame

# In[40]:


import pandas as pd
import nltk
from sklearn.metrics import pairwise_distances
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize
import re
import spacy
from nltk.corpus import stopwords

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[4]:


text = '''QQQ
Invesco QQQ Trust Series 1
Invesco Expands QQQ Innovation Suite to Include Small-Cap ETF
PR Newswire • 4 hours ago
$265.62
1.13%
add_circle_outline
AAPL
Apple Inc
Estimating The Fair Value Of Apple Inc. (NASDAQ:AAPL)
Yahoo Finance • 4 hours ago
$140.41
1.50%
add_circle_outline
TSLA
Tesla Inc
Could This Tesla Stock Unbalanced Iron Condor Return 23%?
Investor's Business Daily • 1 hour ago
$218.30
0.49%
add_circle_outline
AMZN
Amazon.com, Inc.
The Regulators of Facebook, Google and Amazon Also Invest in the Companies' Stocks
Wall Street Journal • 2 days ago
$110.91
1.76%
add_circle_outline'''


# In[5]:


def extract(text):
    
    result = None
    # add your code
    
    return result


# In[75]:


# test your function

extract(text)


# ## Q2: Analyze a document
# 
# When you have a long document, you would like to 
# - Quanitfy how `concrete` a sentence is
# - Create a concise summary while preserving it's key information content and overall meaning. Let's implement an `extractive method` based on the concept of TF-IDF. The idea is to identify the key sentences from an article and use them as a summary. 
# 
# 
# Carefully follow the following steps to achieve these two targets.

# ### Q2.1. Preprocess the input document 
# 
# Define a function `proprocess(doc, lemmatized = True, remove_stopword = True, lower_case = True, remove_punctuation = True, pos_tag = False)` 
# - Four input parameters:
#     - `doc`: an input string (e.g. a document)
#     - `lemmatized`: an optional boolean parameter to indicate if tokens are lemmatized. The default value is True (i.e. tokens are lemmatized).
#     - `remove_stopword`: an optional boolean parameter to remove stop words. The default value is True, i.e., remove stop words. 
#     - `remove_punctuation`: optional boolean parameter to remove punctuations. The default values is True, i.e., remove all punctuations.
#     - `lower_case`: optional boolean parameter to convert all tokens to lower case. The default option is True, i.e., lowercase all tokens.
#     - `pos_tag`: optional boolean parameter to add a POS tag for each token. The default option is False, i.e., no POS tagging.  
#     
#        
# - Split the input `doc` into sentences. Hint, typically, `\n\n+` is used to separate paragraphs. Make sure a sentence does not cross over two paragraphs. You can replace `\n\n+` by a `.`
# 
# 
# - Tokenize each sentence into unigram tokens and also process the tokens as follows:
#     - If `lemmatized` is True, lemmatize all unigrams. 
#     - If `remove_stopword` is set to True, remove all stop words. 
#     - If `remove_punctuation` is set to True, remove all punctuations. 
#     - If `lower_case` is set to True, convert all tokens to lower case 
#     - If `pos_tag` is set to True, find the POS tag for each token and form a tuple for each token, e.g., ('recently', 'ADV'). Either Penn tags or Universal tags are fine. See mapping of these two tagging systems here: https://universaldependencies.org/tagset-conversion/en-penn-uposf.html
# 
# 
# - Return the original sentence list (`sents`) and also the tokenized (or tagged) sentence list (`tokenized_sents`). 
# 
#    
# (Hint: you can use [nltk](https://www.nltk.org/api/nltk.html) and [spacy](https://spacy.io/api/token#attributes) package for this task.)

# In[151]:


nlp = spacy.load("en_core_web_sm")
    
def preprocess(doc, lemmatized=True, pos_tag = False, remove_stopword=True, lower_case = True, remove_punctuation = True):
    
    sents, tokenized_sents = None, None
    text1=re.sub(r'\n\n+','.',text)
    sents= [i for i in nlp(text1).sents]
    
    tokens=[]
    for i in sents:
        tokens.append([j.text for j in i])
        
    if lemmatized:
        lemma=[]
        for i in sents:
            lemma.append([j.lemma_.lower() for j in i if not j.is_punct and j.lemma_ not in stop_words])
        tokenized_sents=lemma
    
    if pos_tag:
        pos=[]
        for i in sents:
            pos.append([(j.text,j.pos_) for j in i])
        tokenized_sents=pos
        
        
    return sents ,tokenized_sents


# In[42]:


# load test document

text = open("power_of_nlp.txt", "r", encoding='utf-8').read()


# In[83]:


# test with all default options:

sents, tokenized_sents = preprocess(text)

# print first 3 sentences
for i in range(3):
    print(sents[i], "\n",tokenized_sents[i],"\n\n" )


# In[104]:


# process text without remove stopwords, punctuation, lowercase, but with pos tagging

sents, tokenized_sents = preprocess(text, lemmatized = False, pos_tag = True, 
                                    remove_stopword=False, remove_punctuation = False, 
                                    lower_case = False)

for i in range(3):
    print(sents[i], "\n",tokenized_sents[i],"\n\n" )


# ### Q2.2. Quantify sentence concreteness
# 
# 
# `Concreteness` can increase a message's persuasion. The concreteness can be measured by the use of :
# - `article` (e.g., a, an, and the), 
# - `adpositions` (e.g., in, at, of, on, etc), and
# - `quantifiers`, i.e., adjectives before nouns.
# 
# 
# Define a function `compute_concreteness(tagged_sent)` as follows:
# - Input argument is `tagged_sent`, a list with (token, pos_tag) tuples as shown above.
# - Find the three types of tokens: `articles`, `adposition`, and `quantifiers`.
# - Compute `concereness` score as:  `(the sum of the counts of the three types of tokens)/(total non-punctuation tokens)`.
# - return the concreteness score, articles, adposition, and quantifiers lists.
# 
# 
# Find the most concrete and the least concrete sentences from the article. 
# 
# 
# Reference: Peer to Peer Lending: The Relationship Between Language Features, Trustworthiness, and Persuasion Success, https://socialmedialab.sites.stanford.edu/sites/g/files/sbiybj22976/files/media/file/larrimore-jacr-peer-to-peer.pdf

# In[85]:


def compute_concreteness(tagged_sent):
    
    #concreteness, articles, adpositions,quantifier = None, None, None, None
    
    article=[]
    adposition=[]
    quantifiers=[]
    article_count=0
    adpos_count=0
    quanti=0
    non=0
    for i in x:
        if i[1]=='DET':
            article.append(i)
            article_count+=1
        if i[1]=='ADP':
            adposition.append(i)
            adpos_count+=1
        if i[1]=='ADJ':
            quantifiers.append(i)
            quanti+=1
    for i in x:
        if i[1]!='PUNCT':
            non+=1
    
    concreteness=(article_count+adpos_count+quanti)/non
    articles=article
    adpositions=adposition
    quantifier=quantifiers
    
    return concreteness, articles, adpositions,quantifier
    


# In[110]:


# tokenize with pos tag, without change the text much

sents, tokenized_sents = preprocess(text, lemmatized = False, pos_tag = True, 
                                    remove_stopword=False, remove_punctuation = False, 
                                    lower_case = False)


# In[108]:


# find concreteness score, articles, adpositions, and quantifiers in a sentence

idx = 1    # sentence id
x = tokenized_sents[idx]
concreteness, articles, adpositions,quantifier = compute_concreteness(x)

# show sentence
sents[idx]
# show result
concreteness, articles, adpositions,quantifier


# In[121]:


# Find the most concrete and the least concrete sentences from the article


    


print (f"The most concerete sentence:  {sents[max_id]}, {concrete[max_id]:.3f}\n")
print (f"The least concerete sentence:  {sents[min_id]}, {concrete[min_id]:.3f}")


# ### Q2.3. Generate TF-IDF representations for sentences 
# 
# Define a function `compute_tf_idf(sents, use_idf)` as follows: 
# 
# 
# - Take the following two inputs:
#     - `sents`: tokenized sentences (without pos tagging) returned from Q2.1. These sentences form a corpus for you to calculate `TF-IDF` vectors.
#     - `use_idf`: if this option is true, return smoothed normalized `TF_IDF` vectors for all sentences; otherwise, just return normalized `TF` vector for each sentence.
#     
#     
# - Calculate `TF-IDF` vectors as shown in the lecture notes (Hint: you can slightly modify code segment 7.5 in NLP Lecture Notes (II) for this task)
# 
# - Return the `TF-IDF` vectors  if `use_idf` is True.  Return the `TF` vectors if `use_idf` is False.

# In[178]:


def compute_tf_idf(sents, use_idf = True):
    
    tf_idf = None
    token_count={}
    for i in range(len(tokenized_sents)):
        token_count[i]=nltk.FreqDist(tokenized_sents[i])
    dtm=pd.DataFrame.from_dict(token_count,orient="index" )
    dtm=dtm.fillna(0)
    dtm = dtm.sort_index(axis = 0)
    tf=dtm.values
    doc_len=tf.sum(axis=1)
    tf=np.divide(tf, doc_len[:,None])
    df=np.where(tf>0,1,0)
    smoothed_idf=np.log(np.divide(len(sents)+1, np.sum(df, axis=0)+1))+1
    
    if use_idf:
        tf_idf=normalize(tf*smoothed_idf)
    else:
        tf_idf=normalize(tf)
    
          
    return tf_idf


# In[99]:


# test compute_tf_idf function

sents, tokenized_sents = preprocess(text)
tf_idf = compute_tf_idf(tokenized_sents, use_idf = True)

# show shape of TF-IDF
tf_idf.shape


# ### Q2.4. Identify key sentences as summary 
# 
# The basic idea is that, in a coherence article, all sentences should center around some key ideas. If we can identify a subset of sentences, denoted as $S_{key}$, which precisely capture the key ideas,  then $S_{key}$ can be used as a summary. Moreover, $S_{key}$ should have high similarity to all the other sentences on average, because all sentences are centered around the key ideas contained in $S_{key}$. Therefore, we can identify whether a sentence belongs to $S_{key}$ by its similarity to all the other sentences.
# 
# 
# Define a function `get_summary(tf_idf, sents, topN = 5)`  as follows:
# 
# - This function takes three inputs:
#     - `tf_idf`: the TF-IDF vectors of all the sentences in a document
#     - `sents`: the original sentences corresponding to the TF-IDF vectors
#     - `topN`: the top N sentences in the generated summary
# 
# - Steps:
#     1. Calculate the cosine similarity for every pair of TF-IDF vectors 
#     1. For each sentence, calculate its average similarity to all the others 
#     1. Select the sentences with the `topN` largest average similarity 
#     1. Print the `topN` sentences index
#     1. Return these sentences as the summary

# In[94]:


def get_summary(tf_idf, sents, topN = 5):
    
    summary = None
    
    
    
    
    return summary 


# In[ ]:





# In[100]:


# put everything together and test with different options

sents, tokenized_sents = preprocess(text)
tf_idf = compute_tf_idf(tokenized_sents, use_idf = True)
summary = get_summary(tf_idf, sents, topN = 5)

for sent in summary:
    print(sent,"\n")


# In[ ]:


# Please test summary generated under different configurations


# ### Q2.5. Analysis 
# 
# - Do you think the way to quantify concreteness makes sense? Any other thoughts to measure concreteness or abstractness? Share your ideas in pdf.
# 
# 
# - Do you think this method is able to generate a good summary? Any pros or cons have you observed? 
# 
# 
# - Do these options `lemmatized, remove_stopword, remove_punctuation, use_idf` matter? 
# - Why do you think these options matter or do not matter? 
# - If these options matter, what are the best values for these options?
# 
# 
# Write your analysis as a pdf file. Be sure to provide some evidence from the output of each step to support your arguments.

# ### Q2.5. (Bonus 3 points). 
# 
# 
# - Can you think a way to improve this extractive summary method? Explain the method you propose for improvement,  implement it, use it to generate a new summary, and demonstrate what is improved in the new summary.
# 
# 
# - Or, you can research on some other extractive summary methods and implement one here. Compare it with the one you implemented in Q2.1-Q2.3 and show pros and cons of each method.

# ## Main block to test all functions

# In[ ]:


if __name__ == "__main__":  
    
    
    text=text = '''QQQ
Invesco QQQ Trust Series 1
Invesco Expands QQQ Innovation Suite to Include Small-Cap ETF
PR Newswire • 4 hours ago
$265.62
1.13%
add_circle_outline
AAPL
Apple Inc
Estimating The Fair Value Of Apple Inc. (NASDAQ:AAPL)
Yahoo Finance • 4 hours ago
$140.41
1.50%
add_circle_outline
TSLA
Tesla Inc
Could This Tesla Stock Unbalanced Iron Condor Return 23%?
Investor's Business Daily • 1 hour ago
$218.30
0.49%
add_circle_outline
AMZN
Amazon.com, Inc.
The Regulators of Facebook, Google and Amazon Also Invest in the Companies' Stocks
Wall Street Journal • 2 days ago
$110.91
1.76%
add_circle_outline'''
    
    
    print("\n==================\n")
    print("Test Q1")
    print(extract(text))
    
    print("\n==================\n")
    print("Test Q2.1")
    
    text = open("power_of_nlp.txt", "r", encoding='utf-8').read()
    
    sents, tokenized_sents = preprocess(text, lemmatized = False, pos_tag = True, 
                                    remove_stopword=False, remove_punctuation = False, 
                                    lower_case = False)
    
    idx = 1    # sentence id
    x = tokenized_sents[idx]
    concreteness, articles, adpositions,quantifier = compute_concreteness(x)

    # show sentence
    sents[idx]
    # show result
    concreteness, articles, adpositions,quantifier
    
    print("\n==================\n")
    print("Test Q2.2-2.4")
    sents, tokenized_sents = preprocess(text)
    tf_idf = compute_tf_idf(tokenized_sents, use_idf = True)
    summary = get_summary(tf_idf, sents, topN = 5)
    print(summary)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




