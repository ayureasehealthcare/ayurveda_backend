from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI(
    title="Llama3.2 11B Vision-Instruct + Ayurveda LoRA API",
    description="Serve vision-base Llama 3.2-11B model finetuned with text LoRA adapter.",
    version="1.0"
)

model = None
tokenizer = None


class Query(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7


@app.on_event("startup")
async def load_lora_model():
    global model, tokenizer
    base_model_id = "unsloth/llama-3.2-11b-vision-instruct-bnb-4bit"
    lora_adapter_id = "ayureasehealthcare/llama3-ayurveda-lora-v3"
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )
        model_lora = PeftModel.from_pretrained(base_model, lora_adapter_id)
        model = model_lora
    except Exception as e:
        print(f"Error loading model or adapter: {e}")


@app.get("/")
def root():
    return {"message": "Llama3 Vision-Instruct 11B with Ayurveda LoRA API online."}


@app.get("/health")
def health():
    return {"ok": model is not None and tokenizer is not None}


@app.post("/ask")
async def ask(query: Query):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    prompt = query.prompt
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=query.max_tokens,
                temperature=query.temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
    except RuntimeError as err:
        # Handle low-memory errors gracefully
        import gc
        gc.collect()
        raise HTTPException(status_code=500, detail=f"Runtime error during generation: {err}")

    out_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if out_text.startswith(prompt):
        answer = out_text[len(prompt):].strip()
    else:
        answer = out_text.strip()

    return {"response": answer, "prompt": prompt}
  
