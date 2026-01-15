import re


def heuristic_extract(text: str):
    fields = {}

    inv = re.search(r"(invoice\s*#?\s*\d+|\b\d{6,}\b)", text, re.I)
    if inv:
        fields["INVOICE_NO"] = inv.group().split()[-1]

    date = re.search(r"\b\d{1,2}/\d{1,2}/\d{4}\b", text)
    if date:
        fields["DATE"] = date.group()

    amt = re.findall(r"\$?\s?\d+[.,]\d{2}", text)
    if amt:
        fields["TOTAL_AMOUNT"] = amt[-1]

    vendor = re.search(r"^[A-Z][A-Za-z &,.]{5,}", text)
    if vendor:
        fields["VENDOR"] = vendor.group().strip()

    return fields
