#%%
import flair
print(flair.__version__)

from flair.data import Dictionary, Sentence
from flair.models import SequenceTagger


#%% Modify the loss function and evaluate function of student model 
# modified self.forward, which leads to self.forward_loss, and modified _calculate_loss

#%%
from flair.models import SequenceTagger

from flair.data import Corpus
from flair.datasets import UD_ENGLISH
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings

# 1. get the corpus
corpus: Corpus = UD_ENGLISH().downsample(0.1)
print(corpus)

# 2. what tag do we want to predict?
tag_type = 'pos'

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
print(tag_dictionary)

# 4. initialize embeddings
embedding_types = [

    WordEmbeddings('glove'),

    # comment in this line to use character embeddings
    # CharacterEmbeddings(),

    # comment in these lines to use flair embeddings
    # FlairEmbeddings('news-forward'),
    # FlairEmbeddings('news-backward'),
]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type,
                                        KD=True)

# 6. initialize trainer
from flair.trainers import ModelTrainer

trainer: ModelTrainer = ModelTrainer(tagger, corpus)
#%%
# 7. start training
trainer.train('resources/taggers/example-pos',
              learning_rate=0.1,
              mini_batch_size=5,
              max_epochs=2)


# %%
sent = corpus.get_all_sentences()[3]
for token in sent:

    print(token.get_tag('pos').value)

#%%
corpus.test[3].get_spans('pos')
# %%
orig_sent = Sentence(corpus.test[3].to_original_text())
print(orig_sent.get_spans('pos'))
tagger.predict(orig_sent)
print(orig_sent.get_spans('pos'))


#%%
import flair
print(flair.__version__)

from flair.data import Sentence
from flair.models import SequenceTagger

# load tagger
tagger = SequenceTagger.load("flair/ner-german-large")
# %%
# import news data
import json
import numpy as np
sentences = json.load(open("/Users/Wu/Google Drive/NER/sentences.json", 'r', encoding='utf-8'))

print(np.histogram([len(s) for s in sentences]))

from flair.datasets import CONLL_03_GERMAN
corpus = CONLL_03_GERMAN(base_path = '/Users/Wu/Google Drive/',encoding= 'latin-1' )

print(len(sentences))
print(len(corpus.train))
print(len(corpus.dev))
# %%
from sequence_tagger_model_KD import SequenceTagger
from flair.data import Sentence
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings
from flair.trainers import ModelTrainer

tag_type = 'ner'
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
print(tag_dictionary)
embedding_types = [
    WordEmbeddings('glove'),
]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type,
                                        KD=True)
                                        
trainer: ModelTrainer = ModelTrainer(tagger, corpus)

sentence = Sentence(sentences[0])
tagger.predict(sentence,all_tag_prob=True)
#%%
span_list = sentence.get_spans('ner')
span = span_list[0]
token = span[0]
dist = token.get_tags_proba_dist('ner')
dist[5].score 
#TODO: however, print(dist) shows error

from tqdm import tqdm
from flair.data import Corpus
data_train = []
for sentence in tqdm(sentences[0:800]):
  sentence = Sentence(sentence)
  tagger.predict(sentence,all_tag_prob=True)
  # data_train.extend(LabeledString(sentence))
  data_train.append(sentence)

import pickle
with open('/Users/Wu/Google Drive/result_prob8.pickle', 'wb') as handle:
    pickle.dump(data_train, handle, protocol=pickle.HIGHEST_PROTOCOL)


# %%
from flair.models import SequenceTagger
from flair.data import Sentence
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings
from flair.trainers import ModelTrainer

tag_type = 'ner'

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
print(tag_dictionary)

# 4. initialize embeddings
embedding_types = [
    WordEmbeddings('glove'),
]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type,
                                        KD=True)
                                        
trainer: ModelTrainer = ModelTrainer(tagger, corpus)

sentence = Sentence(sentences[0])
tagger.predict(sentence,all_tag_prob=True)

#%%
from tqdm import tqdm
from flair.data import Corpus
data_train = []
for sentence in tqdm(sentences[0:800]):
  sentence = Sentence(sentence)
  tagger.predict(sentence,all_tag_prob=True)
  # data_train.extend(LabeledString(sentence))
  data_train.append(sentence)

# import pickle
# with open('/Users/Wu/Google Drive/result_prob7.pickle', 'wb') as handle:
#     pickle.dump(data_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
#%%
span_list = sentence.get_spans('ner')
span = span_list[0]
token = span[0]
dist = token.get_tags_proba_dist('ner')
dist[5].score
