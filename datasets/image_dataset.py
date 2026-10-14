from .multimodal_dataset import MultimodalDataset

class ImageDataset(MultimodalDataset):
    def __init__(self, *args, num_classes=2, **kwargs):
        super().__init__(*args, **kwargs)
        if any(r["task"] != "image" or r["label"] >= num_classes for r in self.rows):
            raise ValueError("Image dataset requires image tasks and labels within num_classes")
