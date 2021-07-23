#%% ###############first part : load tagger and predict data, save the distribution result######################
import flair
print(flair.__version__)

# from flair.models import SequenceTagger
from sequence_tagger_model_KD import SequenceTagger

teacher = SequenceTagger.load("flair/ner-german-large")

# import data
import json
import numpy as np
sentences_news = json.load(open("/Users/Wu/Google Drive/NER/sentences.json", 'r', encoding='utf-8'))

print(np.histogram([len(s) for s in sentences_news]))

from flair.datasets import CONLL_03_GERMAN
corpus = CONLL_03_GERMAN(base_path = '/Users/Wu/Google Drive/',encoding= 'latin-1' )

print(len(sentences_news))
print(len(corpus.train))
print(len(corpus.dev))

sentences_conll = []
for sent in corpus.train:
    sentences_conll.append(sent.to_original_text())
#%% select Nr of sentences in each dataset and mix them 
Nr = 1000
sentences = sentences_conll[0:Nr] + sentences_news[0:Nr]
#%%
# predict data with ner_large 
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
with open('/Users/Wu/Google Drive/data.pickle', 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

# #%%
# before_predict = corpus.train[6].get_spans('ner')#[0][0].get_tags_proba_dist('ner')
# before_predict
# # %%
# after_predict = data[16].get_spans('ner')[0][0].get_tags_proba_dist('ner')
# after_predict

#%% ##############################second part: load data, and train the model############################ 
import pickle
from flair.data import Corpus
from flair.datasets import SentenceDataset
from sklearn.model_selection import train_test_split
from sequence_tagger_model_KD import SequenceTagger
from flair.embeddings import TokenEmbeddings, WordEmbeddings, CharacterEmbeddings,StackedEmbeddings
from flair.embeddings import TransformerWordEmbeddings

with open('/Users/Wu/Google Drive/data.pickle', 'rb') as handle:
    data = pickle.load(handle)
data_train, data_test = train_test_split(data, test_size=0.2, random_state=42)
data_test, data_dev = train_test_split(data_test, test_size=0.5, random_state=42)
corpus: Corpus = Corpus(SentenceDataset(data_train),SentenceDataset(data_test),SentenceDataset(data_dev))
# from flair.datasets import UD_ENGLISH
# corpus: Corpus = UD_ENGLISH().downsample(0.1)
tag_type = 'ner'
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

# embeddings = TransformerWordEmbeddings(
#     model='xlm-roberta-large',
#     layers="-1",
#     subtoken_pooling="first",
#     fine_tune=False, # NOTE: ner-large set True
#     use_context=True,
# )
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
                                        KD=True,
                                        )

from flair.trainers import ModelTrainer

trainer: ModelTrainer = ModelTrainer(tagger, corpus)
#%% train the model 
trainer.train('resources/taggers/example-ner',
              learning_rate=0.1,
              mini_batch_size=10,
              max_epochs=5,
              checkpoint=True,
              )
#%% resume training
checkpoint = 'resources/taggers/example-ner/checkpoint.pt'
trainer = ModelTrainer.load_checkpoint(checkpoint, corpus) #NOTE: ner-large uses optimizer=torch.optim.AdamW
trainer.train('resources/taggers/example-ner',
              learning_rate=0.1,
              mini_batch_size=10,
              max_epochs=20,
              checkpoint=True) 

#NOTE: the trainer of ner-large
# trainer.train('resources/taggers/ner-german-large',
#               learning_rate=5.0e-6,
#               mini_batch_size=4,
#               mini_batch_chunk_size=1,
#               max_epochs=20,
#               scheduler=OneCycleLR,
#               embeddings_storage_mode='none',
#               weight_decay=0.,
#               )

#%% ###############third part: loaded trained model and evaluate on testset or external dataset###############
teacher = SequenceTagger.load("flair/ner-german-large")
student = SequenceTagger.load('resources/taggers/example-ner/best-model.pt')
#%% # first test on data_test: f1 scores 
result_t, eval_loss_t = teacher.evaluate(corpus.test[0:100],mini_batch_size=10)
result_s, eval_loss_s = student.evaluate(corpus.test[0:100],mini_batch_size=10)

print(result_t.main_score)
print(result_s.main_score)




#%%#########################################################################################################
# corpus = CONLL_03_GERMAN(base_path = '/Users/Wu/Google Drive/',encoding= 'latin-1' )
data_train[1].get_spans('ner')[0][0].get_tags_proba_dist('ner')
# corpus.train[2].get_spans('ner')[0][0]#.get_tags_proba_dist('ner')
# data_dev[11].get_spans('ner')[0][0].get_tags_proba_dist('ner')
from flair.data import Sentence
exp = Sentence("George Washington ging nach Washington")
student.predict(exp)
exp.get_spans('ner')
