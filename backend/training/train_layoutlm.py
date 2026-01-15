import torch
from torch.utils.data import DataLoader
from transformers import LayoutLMv3ForTokenClassification
from backend.training.layoutlm_dataset import LayoutLMDataset
from backend.config.labels import LABEL2ID, ID2LABEL


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LayoutLMv3ForTokenClassification.from_pretrained(
        "microsoft/layoutlmv3-base",
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID
    ).to(device)

    dataset = LayoutLMDataset(
        json_path="data/outputs/layoutlm_train.json"
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    model.train()

    for epoch in range(5):
        total_loss = 0.0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Loss: {total_loss:.4f}")

    model.save_pretrained("models/layoutlm_invoice")
    print("✅ Model saved to models/layoutlm_invoice")


if __name__ == "__main__":
    train()
