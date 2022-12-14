from pathlib import Path
from typing import List
from flair.datasets import ColumnCorpus
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.data import Dictionary, Sentence
from flair.trainers import ModelTrainer
from flair.tokenization import SegtokSentenceSplitter


class NERTagger:
    def __init__(
        self,
        label_dict: Dictionary,
        model: str = "bert-base-uncased",
        layers: str = "-1",
        subtoken_pooling: str = "first",
        tagger_hidden_size: int = 256,
    ) -> None:
        embeddings = TransformerWordEmbeddings(
            model=model,
            layers=layers,
            subtoken_pooling=subtoken_pooling,
            fine_tune=True,
            use_context=True,
        )

        self.tagger = SequenceTagger(
            hidden_size=tagger_hidden_size,
            embeddings=embeddings,
            tag_dictionary=label_dict,
            tag_type="ner",
            use_crf=False,
            use_rnn=False,
            reproject_embeddings=False,
        )

    @classmethod
    def init_from_load(cls, path : str | Path):
        obj = cls.__new__(cls)
        obj.tagger = SequenceTagger.load(path)
        
        return obj
    
    def train(
        self,
        corpus: ColumnCorpus,
        save_path: str,
        learning_rate: float = 5.0e-6,
        mini_batch_size: int = 4,
        mini_batch_chunk_size: int | None = None,
        checkpoint: bool = False,
        **kwargs
    ) -> dict:
        trainer = ModelTrainer(self.tagger, corpus)
        
        results = trainer.fine_tune(
            base_path=save_path,
            learning_rate=learning_rate,
            mini_batch_size=mini_batch_size,
            mini_batch_chunk_size=mini_batch_chunk_size,
            checkpoint=checkpoint,
            **kwargs
        )

        return results

    def predict(self, text: str) -> List[Sentence]:
        splitter = SegtokSentenceSplitter()
        sentences = splitter.split(text)
        
        self.tagger.predict(sentences)

        return sentences

    def load(self, save_path: str) -> None:
        self.tagger = SequenceTagger.load(save_path)

    def resume(self, save_path: str, max_epochs: int) -> None:
        self.load(save_path + "/checkpoint.pt")
        self.trainer.resume(
            self.tagger,
            base_path=save_path + "-resume",
            max_epochs=max_epochs,
        )
