#%% ##############################second part: load data, and train the model############################ 
import pickle
from flair.data import Corpus
from flair.datasets import SentenceDataset
from sklearn.model_selection import train_test_split
from sequence_tagger_model_KD import SequenceTagger
from flair.embeddings import TokenEmbeddings, WordEmbeddings, CharacterEmbeddings,StackedEmbeddings
from flair.embeddings import TransformerWordEmbeddings
import pickle5 as pickle
# path = '/Users/Wu/Google Drive/'
path=''
with open(path+'data/data_25k.pickle', 'rb') as handle:
    data = pickle.load(handle)
data_train, data_test = train_test_split(data, test_size=0.2, random_state=42)
data_dev,data_test = train_test_split(data_test, test_size=0.5, random_state=42)
corpus: Corpus = Corpus(SentenceDataset(data_train),SentenceDataset(data_test),SentenceDataset(data_dev))
# corpus: Corpus = corpus.downsample(0.1)
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
trainer.train('resources/taggers/ner_KD_25k',
              learning_rate=0.1,
              mini_batch_size=10,
              max_epochs=30,
              checkpoint=True,
              )
#%% resume training
# checkpoint = 'resources/taggers/ner_KD_25k_30ep/checkpoint.pt'
# trainer = ModelTrainer.load_checkpoint(checkpoint, corpus) #NOTE: ner-large uses optimizer=torch.optim.AdamW
# trainer.train('resources/taggers/ner_KD_25k_50ep',
#               learning_rate=0.02,
#               mini_batch_size=10,
#               max_epochs=50,
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
# teacher = SequenceTagger.load("flair/ner-german-large")
# student = SequenceTagger.load('resources/taggers/example-ner/best-model.pt')


