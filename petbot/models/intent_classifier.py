from pathlib import Path
from typing import List
from flair.models import TextClassifier
from flair.datasets import ClassificationCorpus
from flair.data import Dictionary, Sentence
from flair.embeddings import TransformerDocumentEmbeddings
from flair.trainers import ModelTrainer
from flair.tokenization import SegtokSentenceSplitter


class IntentClassifier:
    def __init__(
        self,
        label_dict: Dictionary,
        model: str = "bert-base-uncased",
    ) -> None:
        embeddings = TransformerDocumentEmbeddings(
            model=model,
            fine_tune=True,
        )

        self.classifier = TextClassifier(
            embeddings,
            label_dictionary=label_dict,
            label_type="intent",
        )
    
    @classmethod
    def init_from_load(cls, path : str | Path):
        obj = cls.__new__(cls)
        obj.classifier = TextClassifier.load(path)
        
        return obj
    
    def train(
        self,
        corpus: ClassificationCorpus,
        save_path: str,
        learning_rate: float = 5.0e-5,
        mini_batch_size: int = 4,
        max_epochs: int = 10,
        checkpoint: bool = False,
        **kwargs
    ) -> dict:
        trainer = ModelTrainer(self.classifier, corpus)
        
        results = trainer.train(
            save_path,
            learning_rate=learning_rate,
            mini_batch_size=mini_batch_size,
            max_epochs=max_epochs,
            checkpoint=checkpoint,
            **kwargs
        )

        return results

    def predict(self, text: str) -> List[Sentence]:
        splitter = SegtokSentenceSplitter()
        sentences = splitter.split(text)
        
        self.classifier.predict(sentences)
        
        return sentences

    def resume(self, save_path: str, max_epochs: int) -> None:
        self.load(save_path + "/checkpoint.pt")
        self.trainer.resume(
            self.classifier,
            base_path=save_path + "-resume",
            max_epochs=max_epochs,
        )
