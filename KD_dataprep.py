#%% ###############first part : load tagger and predict data, save the distribution result######################

# path = '/Users/Wu/Google Drive/'
path = '/content/gdrive/My Drive/'
import flair
print(flair.__version__)

# from flair.models import SequenceTagger
from sequence_tagger_model_KD import SequenceTagger

teacher = SequenceTagger.load("flair/ner-german-large")

# import data
import json
import numpy as np
sentences_news = json.load(open(path+'data/sentences.json', 'r', encoding='utf-8'))

print(np.histogram([len(s) for s in sentences_news]))

from flair.datasets import CONLL_03_GERMAN
corpus = CONLL_03_GERMAN(base_path = path ,encoding= 'latin-1' ) 
#NOTE: path should contain folder conll_03_german, which contains the conll dataset there.

print('------------------------------')
print('Nr of sentences in news data: ',len(sentences_news))
print('Nr of sentences in conll training set: ',len(corpus.train))
print('Nr of sentences in conll development set: ',len(corpus.dev))

sentences_conll = []
for sent in corpus.train:
    sentences_conll.append(sent.to_original_text()) 
#since the conll dataset has tags, but we don't need the real hard label, but the predicted soft ones by teacher model
#%% select Nr of sentences in each dataset and mix them 
Nr = len(corpus.train)
sentences = sentences_conll[0:Nr] + sentences_news[0:Nr]

#%% # predict data with ner_large 
from flair.data import Sentence
from tqdm import tqdm
data = []
for sentence in tqdm(sentences):
    if sentence != '':
        sentence = Sentence(sentence)
        teacher.predict(sentence,all_tag_prob=True)
        data.append(sentence)

#%% save the data(together with the distribution inside the object) to trainset
import pickle
with open(path+'data/data_25k.pickle', 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)