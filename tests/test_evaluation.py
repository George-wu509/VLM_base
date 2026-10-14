import pytest
from evaluation.caption import rouge_l
from evaluation.vqa import exact_match
from evaluation.grounding import grounding_metrics, iou
from evaluation.hallucination import score_yes_no
from tokenization.spatial_tokens import encode_box
from scripts.promote import check_gates

def test_caption_and_vqa_metrics():
    assert rouge_l("a red rectangle", "a red rectangle") == 1
    assert rouge_l("blue", "red") == 0
    assert exact_match("The BLUE!", "blue") == 1

def test_grounding_matching_penalizes_extra_predictions():
    truth = [{"bbox": [0.1, 0.1, 0.9, 0.9], "label": "rectangle"}]
    prediction = encode_box(truth[0]["bbox"])+" rectangle ; "+encode_box([0.2, 0.2, 0.3, 0.3])+" other"
    result = grounding_metrics(prediction, truth)
    assert result["grounding_recall_at_0_5"] == 1
    assert result["grounding_precision_at_0_5"] == 0.5
    assert result["grounding_mean_iou"] > 0.99

def test_hallucination_probe_tracks_invalid_answers():
    result = score_yes_no(["yes", "no", "unclear"], ["no", "yes", "no"])
    assert result["false_positive_rate"] == 0.5
    assert result["presence_recall"] == 0
    assert result["invalid_answer_rate"] == 1/3

def test_promotion_rejects_regression_and_requires_metrics():
    baseline = {"metrics": {"accuracy": 0.8}}
    rules = {"metrics": {"accuracy": {"direction": "higher", "minimum": 0.7, "max_regression": 0.02}}}
    assert check_gates(baseline, {"metrics": {"accuracy": 0.81}}, rules) == []
    assert check_gates(baseline, {"metrics": {"accuracy": 0.5}}, rules)
    assert check_gates(baseline, {"metrics": {}}, rules)
    assert check_gates(baseline, {"metrics": {"accuracy": float("nan")}}, rules)
