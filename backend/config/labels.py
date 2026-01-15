LABELS = [
    "O",
    "INVOICE_NO",
    "DATE",
    "TOTAL_AMOUNT",
    "VENDOR",
]

LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}
