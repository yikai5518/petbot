from flair.models import TextClassifier
from flair.datasets import ClassificationCorpus
from flair.data import Dictionary, Sentence
from flair.embeddings import TransformerDocumentEmbeddings
from flair.trainers import ModelTrainer


class IntentClassifier:
    def __init__(
        self,
        corpus: ClassificationCorpus,
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

        self.trainer = ModelTrainer(self.classifier, corpus)

    def train(
        self,
        save_path: str,
        learning_rate: float = 5.0e-5,
        mini_batch_size: int = 4,
        max_epochs: int = 10,
        checkpoint: bool = False,
        **kwargs
    ) -> dict:
        results = self.trainer.train(
            save_path,
            learning_rate=learning_rate,
            mini_batch_size=mini_batch_size,
            max_epochs=max_epochs,
            checkpoint=checkpoint,
            **kwargs
        )

        return results

    def predict(self, sentence: Sentence) -> str | None:
        self.classifier.predict(sentence)
        return sentence.labels

    def load(self, save_path: str) -> None:
        self.classifier = TextClassifier.load(save_path)

    def resume(self, save_path: str, max_epochs: int) -> None:
        self.load(save_path + "/checkpoint.pt")
        self.trainer.resume(
            self.classifier,
            base_path=save_path + "-resume",
            max_epochs=max_epochs,
        )
