from .multimodal_dataset import MultimodalDataset

class TextDataset(MultimodalDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if any(r["task"] != "text" or r.get("image") for r in self.rows):
            raise ValueError("Text-only pretraining requires text tasks without images")
