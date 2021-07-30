#%% ##############################second part: load data, and train the model############################ 
import pickle5 as pickle
from flair.data import Corpus
from flair.datasets import SentenceDataset
from sklearn.model_selection import train_test_split
from sequence_tagger_model_KD import SequenceTagger
from flair.embeddings import TokenEmbeddings, WordEmbeddings, CharacterEmbeddings,StackedEmbeddings
from flair.embeddings import TransformerWordEmbeddings
#%%
# path = '/Users/Wu/Google Drive/'
path = '/content/gdrive/My Drive/'

with open(path+'data/data_2k.pickle', 'rb') as handle: #NOTE: data_2k.pickle as small sample
    data = pickle.load(handle)
data_train, data_test = train_test_split(data, test_size=0.2, random_state=42)
data_dev,data_test = train_test_split(data_test, test_size=0.5, random_state=42)
corpus: Corpus = Corpus(SentenceDataset(data_train),SentenceDataset(data_test),SentenceDataset(data_dev))
# corpus: Corpus = corpus.downsample(0.1)
tag_type = 'ner'

tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

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
                                        debug=True,
                                        )

from flair.trainers import ModelTrainer

trainer: ModelTrainer = ModelTrainer(tagger, corpus)
#%% train the model 
trainer.train('resources/taggers/ner_25k_30ep',
              learning_rate=0.1,
              mini_batch_size=10,
              max_epochs=30,
              checkpoint=True,
              )
#%% resume training
# checkpoint = 'resources/taggers/ner_KD_25k_5ep/checkpoint.pt'
# trainer = ModelTrainer.load_checkpoint(checkpoint, corpus) #NOTE: ner-large uses optimizer=torch.optim.AdamW
# trainer.train('resources/taggers/ner_KD_25k_30ep',
#               learning_rate=0.1,
#               mini_batch_size=10,
#               max_epochs=30,
#               checkpoint=True) 

