from PIL import Image, ImageDraw

def draw_grounding(image_path, objects, output):
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    w, h = image.size
    for obj in objects:
        x0, y0, x1, y1 = obj["bbox"]
        draw.rectangle((x0*w, y0*h, x1*w, y1*h), outline="red", width=2)
        draw.text((x0*w, max(0, y0*h-12)), obj["label"], fill="red")
    image.save(output)
