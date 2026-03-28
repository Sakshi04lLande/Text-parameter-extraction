from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "bhadresh-savani/bert-base-go-emotion"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)