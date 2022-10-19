from pathlib import Path
from typing import Tuple
import pandas as pd
from flair.datasets import ColumnCorpus, ClassificationCorpus


def clean(text : str) -> str:
    filters = ["!", "#", "$", "%", "&", "(", ")", "/", "*", ".", ":", ";", "<", "=", ">", "?", "@", "[",
               "\\", "]", "_", "`", "{", "}", "~", "'"]
    for i in text:
        if i in filters:
            text = text.replace(i, " " + i)
            
    return text


def transform_ner_data(df: pd.DataFrame, path: str | Path) -> None:
    pass


def transform_intent_data(df: pd.DataFrame, path: str | Path) -> None:
    return


def load_ner_data(path: str | Path) -> ColumnCorpus:
    return ColumnCorpus(
        path,
        {0: "text", 1: "ner"},
        train_file="train.txt",
        test_file="test.txt",
        dev_file="dev.txt",
    )


def load_intent_data(path: str | Path) -> ClassificationCorpus:
    return ClassificationCorpus(
        path,
        label_type="intent",
        train_file="train.txt",
        test_file="test.txt",
        dev_file="dev.txt",
    )


def load_data(path: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pass


if __name__ == "__main__":
    train_data, test_data, dev_data = load_data("data/conversation.csv")
    
    transform_intent_data(train_data, "data/intent/train.txt")
    transform_intent_data(test_data, "data/intent/test.txt")
    transform_intent_data(dev_data, "data/intent/dev.txt")
    
    transform_ner_data(train_data, "data/ner/train.txt")
    transform_ner_data(test_data, "data/ner/test.txt")
    transform_ner_data(dev_data, "data/ner/dev.txt")