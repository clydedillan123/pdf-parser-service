from fastapi import FastAPI, UploadFile, File, Header, HTTPException
import os
import tempfile

app = FastAPI()

# API KEY (set in Render environment variables)
API_KEY = os.getenv("API_KEY", "dev-key")

def parse_pdf(file_bytes):
    # Placeholder parser logic — replace with your real parser
    return {
        "records": [],
        "warnings": [],
        "stats": {"message": "Parser executed successfully"}
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/convert-pdf")
async def convert_pdf(
    file: UploadFile = File(...),
    x_api_key: str = Header(None)
):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    contents = await file.read()

    try:
        result = parse_pdf(contents)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
