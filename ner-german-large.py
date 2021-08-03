import torch

# 1. get the corpus
from flair.datasets import CONLL_03_GERMAN

corpus = CONLL_03_GERMAN()

# 2. what tag do we want to predict?
tag_type = 'ner'

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

# 4. initialize fine-tuneable transformer embeddings WITH document context
from flair.embeddings import TransformerWordEmbeddings

embeddings = TransformerWordEmbeddings(
    model='xlm-roberta-large',
    layers="-1",
    subtoken_pooling="first",
    fine_tune=True,
    use_context=True,
)

# 5. initialize bare-bones sequence tagger (no CRF, no RNN, no reprojection)
from flair.models import SequenceTagger

tagger = SequenceTagger(
    hidden_size=256,
    embeddings=embeddings,
    tag_dictionary=tag_dictionary,
    tag_type='ner',
    use_crf=False,
    use_rnn=False,
    reproject_embeddings=False,
)

# 6. initialize trainer with AdamW optimizer
from flair.trainers import ModelTrainer

trainer = ModelTrainer(tagger, corpus, optimizer=torch.optim.AdamW)

# 7. run training with XLM parameters (20 epochs, small LR)
from torch.optim.lr_scheduler import OneCycleLR

trainer.train('resources/taggers/ner-german-large1',
              learning_rate=5.0e-6,
              mini_batch_size=4,
              mini_batch_chunk_size=1,
              max_epochs=20,
              scheduler=OneCycleLR,
              embeddings_storage_mode='none',
              weight_decay=0.,
              )

)