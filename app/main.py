import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from app.api.endpoints import app as endpoints_app

# The main application
app = FastAPI(title="DermAI Full Pipeline Runner")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the advanced endpoints
app.include_router(endpoints_app.router)

INDEX_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>DermAI Testing Dashboard</title>
    <style>
        body { font-family: 'Inter', sans-serif; background-color: #0f172a; color: #f8fafc; padding: 2rem; max-width: 800px; margin: 0 auto;}
        .card { background: #1e293b; padding: 2rem; border-radius: 12px; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5); }
        h1 { color: #38bdf8; }
        button { background-color: #38bdf8; color: #0f172a; border: none; padding: 10px 20px; font-weight: bold; border-radius: 6px; cursor: pointer; transition: 0.2s; }
        button:hover { background-color: #7dd3fc; }
        input[type="file"] { margin: 20px 0; }
        pre { background: #0f172a; padding: 1rem; border-radius: 8px; overflow-x: auto; color: #a7f3d0; }
    </style>
</head>
<body>
    <div class="card">
        <h1>DermAI - Image Upload Testing</h1>
        <p>1. Select a dummy image to test the pipeline (e.g. any PNG or JPG).</p>
        <p>2. The backend will parse it, process via mock RAG endpoints, and return the Pydantic Prediction output.</p>
        
        <input type="file" id="imageInput" accept="image/*">
        <br>
        <button onclick="submitImage()">Analyze Skincare Image</button>

        <h3 style="margin-top: 2rem;">Analysis Output:</h3>
        <pre id="output">Waiting for upload...</pre>
    </div>

    <script>
        async function submitImage() {
            const input = document.getElementById('imageInput');
            if (input.files.length === 0) {
                alert('Please select an image first!');
                return;
            }

            const formData = new FormData();
            formData.append('file', input.files[0]);

            document.getElementById('output').innerText = 'Processing via AI Pipeline...';

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                document.getElementById('output').innerText = JSON.stringify(data, null, 4);
            } catch (error) {
                document.getElementById('output').innerText = 'Error: ' + error.message;
            }
        }
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    return HTMLResponse(content=INDEX_HTML)

if __name__ == "__main__":
    print("🚀 Starting DermAI Engine on http://localhost:8000")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
