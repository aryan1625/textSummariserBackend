from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from summariser import TextSummarizer
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class TextRequest(BaseModel):
    text: str
    mode: str = "medium"
summariser = TextSummarizer()
@app.post("/summarise")
async def summ(req: TextRequest):
    summary = summariser.summarize(req.text,req.mode)
    return {"summary": summary}

@app.post('/upload-pdf')
async def upload_pdf(file:UploadFile = File(...)):
    from PyPDF2 import PdfReader
    import io
    reader = PdfReader(io.BytesIO(await file.read()))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return {"text": text.strip() or "No Text found in the pdf!"}
