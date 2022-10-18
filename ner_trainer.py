from flair.datasets import ColumnCorpus
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.data import Dictionary
from flair.trainers import ModelTrainer

class NER_Trainer():
    def train(
        corpus: ColumnCorpus,
        label_dict: Dictionary,
        save_path: str,
        model: str = "bert-base-uncased",
        layers: str = "-1",
        subtoken_pooling: str = "first",
        tagger_hidden_size: int = 256,
    ):
        embeddings = TransformerWordEmbeddings(
            model=model,
            layers=layers,
            subtoken_pooling=subtoken_pooling,
            fine_tune=True,
            use_context=True,
        )
        
        tagger = SequenceTagger(
            hidden_size=tagger_hidden_size,
            embeddings=embeddings,
            tag_dictionary=label_dict,
            tag_type='ner',
            use_crf=False,
            use_rnn=False,
            reproject_embeddings=False,
        )
        
        trainer = ModelTrainer(tagger, corpus)
        trainer.fine_tune(
            base_path=save_path,
            learning_rate=5.0e-6,
            mini_batch_size=4,
            mini_batch_chunk_size=1,
        )
        
        
    
    