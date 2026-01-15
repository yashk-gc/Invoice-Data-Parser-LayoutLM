from backend.services.postprocess import extract_fields


def test_total_amount_extraction():
    words = ["Total", "$", "5138.35"]
    labels = ["O", "O", "TOTAL_AMOUNT"]

    fields = extract_fields(words, labels)

    assert "TOTAL_AMOUNT" in fields
    assert fields["TOTAL_AMOUNT"] == "5138.35"
