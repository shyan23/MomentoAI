
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import BartForConditionalGeneration, BartTokenizer

# Initialize the FastAPI app
app = FastAPI()


from transformers import BartForConditionalGeneration, BartTokenizer

# Load the fine-tuned model and tokenizer
model_path = "./fine_tuned_model"
model = BartForConditionalGeneration.from_pretrained(model_path)
tokenizer = BartTokenizer.from_pretrained(model_path)

# Move the model to GPU (if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# Define the input schema
class TextInput(BaseModel):
    text: str

# Define a route for summarization
@app.post("/summarize")
async def summarize(input: TextInput):
    try:
        # Tokenize the input text
        inputs = tokenizer.encode(
            input.text,
            return_tensors="pt",
            max_length=1024,
            truncation=True
        )

        # Generate the summary
        summary_ids = model.generate(
            inputs, 
            max_length=150,  # Adjust the max summary length
            min_length=40,   # Adjust the min summary length
            length_penalty=2.0, 
            num_beams=4,
            early_stopping=True
        )

        # Decode the summary
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return {"summary": summary}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
