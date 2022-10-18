from pathlib import Path
import pandas as pd
from flair.datasets import ColumnCorpus, ClassificationCorpus

def transform_ner_data(path : str | Path) -> None:
    pass

def transform_intent_data(path : str | Path) -> None:
    pass

def load_ner_data(path : str | Path) -> ColumnCorpus:
    return ColumnCorpus(
        path,
        {0 : "text", 1 : "ner"},
        train_file="train.txt",
        test_file="test.txt",
        dev_file="dev.txt",
    )

def load_intent_data(path : str | Path) -> ClassificationCorpus:
    return ClassificationCorpus(
        path,
        label_type="intent",
        train_file="train.txt",
        test_file="test.txt",
        dev_file="dev.txt",
    )