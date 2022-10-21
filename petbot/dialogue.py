from pathlib import Path
from petbot.models.intent_classifier import IntentClassifier

from pathlib import Path

from models import NERTagger, IntentClassifier


def dialogue(input : str, intent_cls : IntentClassifier | str | Path, ner_tagger : NERTagger | str | Path) -> str:
    pass
