import re
from tokenization.spatial_tokens import decode_box

PATTERN = re.compile(r"<loc(\d{4})><loc(\d{4})><loc(\d{4})><loc(\d{4})>\s*([^;<>]+)")

def parse_grounding(text):
    objects = []
    for match in PATTERN.finditer(text):
        try:
            box = decode_box([int(v) for v in match.groups()[:4]])
        except ValueError:
            continue
        label = match.group(5).strip()
        if label:
            objects.append({"bbox": box, "label": label})
    return objects
