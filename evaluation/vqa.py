import re
import string

def normalize(text):
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(w for w in text.split() if w not in ("a", "an", "the"))

def exact_match(prediction, reference):
    return float(normalize(prediction) == normalize(reference))

def answer_consensus(prediction, answers):
    # Explicit lightweight min(matches/3,1) metric, not official VQA leave-one-out.
    return min(1.0, sum(normalize(prediction) == normalize(a) for a in answers) / 3)
