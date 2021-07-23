#%% ###############first part : load tagger and predict data, save the distribution result######################

# path = '/Users/Wu/Google Drive/'
path='./'
import flair
print(flair.__version__)
from sequence_tagger_model_KD import SequenceTagger

#%% ##############################second part: load data, and train the model############################ 
# import pickle
import pickle5 as pickle
from flair.data import Corpus
from flair.datasets import SentenceDataset
from sklearn.model_selection import train_test_split
from sequence_tagger_model_KD import SequenceTagger
from flair.embeddings import TokenEmbeddings, WordEmbeddings, CharacterEmbeddings,StackedEmbeddings
from flair.embeddings import TransformerWordEmbeddings

with open(path+'data/data.pickle', 'rb') as handle:
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
              max_epochs=20,
              checkpoint=True,
              )
#%% resume training
# checkpoint = 'resources/taggers/example-ner/checkpoint.pt'
# trainer = ModelTrainer.load_checkpoint(checkpoint, corpus) #NOTE: ner-large uses optimizer=torch.optim.AdamW
# trainer.train('resources/taggers/example-ner',
#               learning_rate=0.1,
#               mini_batch_size=10,
#               max_epochs=20,
#               checkpoint=True) 

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
# # corpus = CONLL_03_GERMAN(base_path = '/Users/Wu/Google Drive/',encoding= 'latin-1' )
# data_train[1].get_spans('ner')[0][0].get_tags_proba_dist('ner')
# # corpus.train[2].get_spans('ner')[0][0]#.get_tags_proba_dist('ner')
# # data_dev[11].get_spans('ner')[0][0].get_tags_proba_dist('ner')
# from flair.data import Sentence
# exp = Sentence("George Washington ging nach Washington")
# student.predict(exp)
# exp.get_spans('ner')
