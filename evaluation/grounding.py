from inference.grounding_parser import parse_grounding
from .vqa import normalize

def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x1-x0)*max(0, y1-y0)
    area_a, area_b = (a[2]-a[0])*(a[3]-a[1]), (b[2]-b[0])*(b[3]-b[1])
    return intersection / max(1e-12, area_a+area_b-intersection)

def grounding_metrics(prediction, reference, threshold=0.5):
    pred = parse_grounding(prediction)
    truth = reference if isinstance(reference, list) else parse_grounding(reference)
    used, hits, overlaps = set(), 0, []
    for obj in truth:
        candidates = [(iou(obj["bbox"], p["bbox"]), i) for i, p in enumerate(pred)
                      if i not in used and normalize(obj["label"]) == normalize(p["label"])]
        overlap, index = max(candidates, default=(0, -1))
        overlaps.append(overlap)
        if index >= 0 and overlap >= threshold:
            hits += 1
            used.add(index)
    return {"grounding_recall_at_0_5": hits/max(1, len(truth)),
            "grounding_precision_at_0_5": hits/max(1, len(pred)),
            "grounding_mean_iou": sum(overlaps)/max(1, len(truth))}
