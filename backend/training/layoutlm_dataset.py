import json
import torch
from torch.utils.data import Dataset
from transformers import LayoutLMv3Processor
from PIL import Image


class LayoutLMDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.processor = LayoutLMv3Processor.from_pretrained(
            "microsoft/layoutlmv3-base",
            apply_ocr=False
        )

    def __len__(self):
        return 1  # single document

    def __getitem__(self, idx):
        image = Image.open(self.data["image_path"]).convert("RGB")

        encoding = self.processor(
            image,
            self.data["words"],
            boxes=self.data["bboxes"],
            word_labels=self.data["labels"],
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {k: v.squeeze(0) for k, v in encoding.items()}
