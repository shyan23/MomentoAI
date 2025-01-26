from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

tokenizer.save_pretrained("./bart_large_cnn_tokenizer")
model.save_pretrained("./bart_large_cnn_model")