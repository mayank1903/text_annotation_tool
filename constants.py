api = "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"
temperature = 0
max_tokens = 1000
model_engine = "text-davinci-003"
specific_word = "MISSED|HELPLINE"
prompt = """
    convert this text and give the above text by filling the information in the placeholder in the below template.

    template is: sent by {sender name}: {crop} of variety {crop variety} - at {location} on {date} at price {crop price} and arrival quantity {arrival quantity}

    crop, crop variety and location will be of string type and crop price and arrival quantity will be of numeric type
    if crop, crop variety, state and date is not available for any record, mention None.

    Also, if it has more than one row of information, separate each row by ';'

    Fill the placeholders with sender, crop, crop variety, location, date, crop price and arrival quantity respectively."""

Buttons = ["commodityb", "varietyb", "locationb", "buyerb", "priceb"]
