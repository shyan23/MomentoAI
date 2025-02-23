
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import BartForConditionalGeneration, BartTokenizer

# Initialize the FastAPI app
app = FastAPI()

# Load your trained DistilBART model and tokenizer
model_path = "~/Desktop/MomentoAI/fine_tuned_model"  # Update this to your model's directory
tokenizer = BartTokenizer.from_pretrained(model_path)
model = BartForConditionalGeneration.from_pretrained(model_path)

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
