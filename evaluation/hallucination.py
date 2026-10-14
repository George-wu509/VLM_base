from .vqa import normalize

def score_yes_no(predictions, answers):
    """Binary object-presence probe metrics for a POPE-style labeled JSONL."""
    if not predictions or len(predictions) != len(answers):
        raise ValueError("Provide paired nonempty predictions/answers")
    truth = [normalize(a) for a in answers]
    if any(a not in ("yes", "no") for a in truth):
        raise ValueError("Object-presence answers must be yes or no")
    pred = [normalize(p).split()[0] if normalize(p).split() else "invalid" for p in predictions]
    tp = sum(p == "yes" and a == "yes" for p, a in zip(pred, truth))
    fp = sum(p == "yes" and a == "no" for p, a in zip(pred, truth))
    fn = sum(p != "yes" and a == "yes" for p, a in zip(pred, truth))
    negatives = truth.count("no")
    return {"presence_accuracy": sum(p == a for p, a in zip(pred, truth))/len(truth),
            "presence_precision": tp/max(1, tp+fp), "presence_recall": tp/max(1, tp+fn),
            "false_positive_rate": fp/max(1, negatives),
            "invalid_answer_rate": sum(p not in ("yes", "no") for p in pred)/len(pred)}
