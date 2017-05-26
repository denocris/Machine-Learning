import numpy as np
import json


features_data = open("data/data_cleaned/textfeatures0", "r")
target_data = open("data/data_cleaned/starstarget0", "r")
text = [line for line in features_data]
target = [int(line.split()[0]) for line in target_data]

target = [ 1 if target[i] >=3 else 0 for i in range(len(target ))]
dataset = [[text[i], target[i]] for i in range(len(target))]


target_pos = target.count(1)
target_neg = target.count(0)
ratio = float(target_neg) / target_pos

import random

def rebalance(dataset_to_reb, length, ratio):
    reb_dataset = []
    for i in range(length):
        if dataset_to_reb[i][1] == 0:
            reb_dataset.append(dataset_to_reb[i])
        else:
            rnd = random.random()
            if rnd < ratio:
                reb_dataset.append(dataset_to_reb[i])
    return reb_dataset

reb_dataset = rebalance(dataset, len(target), ratio)

reb_text = [ tx for [tx,tg] in reb_dataset]
reb_target = [ tg for [tx,tg] in reb_dataset]

default_stop_words = ['all', 'six', 'less', 'being', 'indeed', 'over', 'move', 'anyway', 'not', 'fifty', 'four', 'own', 'through', 'yourselves', 'go', 'where', 'mill', 'only', 'find', 'before', 'one', 'whose', 'system', 'how', 'somewhere', 'with', 'thick', 'show', 'had', 'enough', 'should', 'to', 'must', 'whom', 'seeming', 'under', 'ours', 'has', 'might', 'thereafter', 'latterly', 'do', 'them', 'his', 'around', 'than', 'get', 'very', 'de', 'none', 'cannot', 'every', 'whether', 'they', 'front', 'during', 'thus', 'now', 'him', 'nor', 'name', 'several', 'hereafter', 'always', 'who', 'cry', 'whither', 'this', 'someone', 'either', 'each', 'become', 'thereupon', 'sometime', 'side', 'two', 'therein', 'twelve', 'because', 'often', 'ten', 'our', 'eg', 'some', 'back', 'up', 'namely', 'towards', 'are', 'further', 'beyond', 'ourselves', 'yet', 'out', 'even', 'will', 'what', 'still', 'for', 'bottom', 'mine', 'since', 'please', 'forty', 'per', 'its', 'everything', 'behind', 'un', 'above', 'between', 'it', 'neither', 'seemed', 'ever', 'across', 'she', 'somehow', 'be', 'we', 'full', 'never', 'sixty', 'however', 'here', 'otherwise', 'were', 'whereupon', 'nowhere', 'although', 'found', 'alone', 're', 'along', 'fifteen', 'by', 'both', 'about', 'last', 'would', 'anything', 'via', 'many', 'could', 'thence', 'put', 'against', 'keep', 'etc', 'amount', 'became', 'ltd', 'hence', 'onto', 'or', 'con', 'among', 'already', 'co', 'afterwards', 'formerly', 'within', 'seems', 'into', 'others', 'while', 'whatever', 'except', 'down', 'hers', 'everyone', 'done', 'least', 'another', 'whoever', 'moreover', 'couldnt', 'throughout', 'anyhow', 'yourself', 'three', 'from', 'her', 'few', 'together', 'top', 'there', 'due', 'been', 'next', 'anyone', 'eleven', 'much', 'call', 'therefore', 'interest', 'then', 'thru', 'themselves', 'hundred', 'was', 'sincere', 'empty', 'more', 'himself', 'elsewhere', 'mostly', 'on', 'fire', 'am', 'becoming', 'hereby', 'amongst', 'else', 'part', 'everywhere', 'too', 'herself', 'former', 'those', 'he', 'me', 'myself', 'made', 'twenty', 'these', 'bill', 'cant', 'us', 'until', 'besides', 'nevertheless', 'below', 'anywhere', 'nine', 'can', 'of', 'your', 'toward', 'my', 'something', 'and', 'whereafter', 'whenever', 'give', 'almost', 'wherever', 'is', 'describe', 'beforehand', 'herein', 'an', 'as', 'itself', 'at', 'have', 'in', 'seem', 'whence', 'ie', 'any', 'fill', 'again', 'hasnt', 'inc', 'thereby', 'thin', 'no', 'perhaps', 'latter', 'meanwhile', 'when', 'detail', 'same', 'wherein', 'beside', 'also', 'that', 'other', 'take', 'which', 'becomes', 'you', 'if', 'nobody', 'see', 'though', 'may', 'after', 'upon', 'most', 'hereupon', 'eight', 'but', 'serious', 'nothing', 'such', 'why', 'a', 'off', 'whereby', 'third', 'i', 'whole', 'noone', 'sometimes', 'well', 'amoungst', 'yours', 'their', 'rather', 'without', 'so', 'five', 'the', 'first', 'whereas', 'once']
list_word_to_remove = ['not']
my_stop_words = [word for word in default_stop_words if word not in list_word_to_remove]

my_stop_words.append('ve')
my_stop_words.append('ll')
my_stop_words.append('got')
my_stop_words.append('know')
my_stop_words.append('15')
my_stop_words.append('30')
my_stop_words.append('20')
my_stop_words.append('50')


# In[6]:

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(reb_text, reb_target, test_size=0.2, random_state=42)


# # Linear Model: Logistic Regression
#
#

# In[13]:

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn import pipeline


def my_pipeline_logreg(thres):
    nsw = np.loadtxt("data/stop_words/my_stop_words_thr_%.2f.txt" %thres, dtype=str)
    pipe = pipeline.Pipeline([
                            ('count_vectorizer', CountVectorizer(stop_words = list(nsw), max_df = 0.9, min_df = 10)),
                            #('count_vectorizer', CountVectorizer(stop_words = my_stop_words, max_df = 0.9, min_df = 10)),
                            ('tf_idf',TfidfTransformer()),
                            ('model',LogisticRegression(penalty='l2',
                                                           dual=False,
                                                           tol=0.0001,
                                                           C=1.0,
                                                           fit_intercept=True,
                                                           intercept_scaling=1,
                                                           class_weight=None,
                                                           random_state=None,
                                                           solver='liblinear',
                                                           max_iter=100,
                                                           multi_class='ovr',
                                                           verbose=0,
                                                           warm_start=False,
                                                           n_jobs=1))
                         ])
    return pipe
# # Parameter Optimisation: Grid Search for Logistic Regression

# In[15]:

from sklearn.model_selection import GridSearchCV

pipe = my_pipeline_logreg(0.5)

gs = GridSearchCV(
                    pipe,
                    {
                        "count_vectorizer__min_df": range(1,10)+range(10,30,5),
                        "model__C": [0.1, 0.5, 1.0, 3.0, 5.0, 10, 20, 50, 80]
                    },
                    cv=2,  # 5-fold cross validation
                    n_jobs=20,  # run each hyperparameter in one of two parallel jobs
                    scoring="accuracy" # what could happen selecting "precision" as scoring measure?
                )

gs.fit(X_train, y_train)

# Saving Data
import cPickle as pickle

with open('lr_grid_results.csv', 'wb') as fp:
    pickle.dump(gs.cv_results_, fp)
    
with open('lr_grid_bestparameters.csv', 'wb') as fp:
    pickle.dump(gs.best_params_, fp)



