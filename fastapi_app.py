import os
import sys
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from SpeechToText.pipeline.training_pipeline import TrainingPipeline
from SpeechToText.pipeline.prediction_pipeline import Prediction
from SpeechToText.entity.config_entity import PredictionPipelineConfig
from SpeechToText.constants import S3_BUCKET_URI

app = FastAPI(
    title="Speech-to-Text API",
    description="FastAPI backend for audio processing.",
    version="1.0.0"
)


templates = Jinja2Templates(directory="template")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # Serve the new interface
    return templates.TemplateResponse(request=request, name="frontend.html")

@app.post("/predict")
async def predict_audio(audio: UploadFile = File(...)):

    config = PredictionPipelineConfig()

    os.makedirs(config.prediction_artifact_dir, exist_ok=True)
    os.makedirs(config.model_download_path, exist_ok=True)
    os.makedirs(config.app_artifacts, exist_ok=True)

    # Sync model from S3
    sync_cmd = f'aws s3 sync "{config.s3_model_path}" "{config.model_download_path}"'
    os.system(sync_cmd)

    wave_sounds_path = config.wave_sounds_path
    
    # Read the audio bytes and save them locally
    contents = await audio.read()
    with open(wave_sounds_path, "wb") as f:
        f.write(contents)

    # Find the downloaded weights file
    weight_files = [f for f in os.listdir(config.model_download_path) if f.endswith('.weights.h5')]
    if weight_files:
        model_path = os.path.join(config.model_download_path, weight_files[0])
    else:
        raise HTTPException(
            status_code=500, 
            detail=f"No .weights.h5 file found in {config.model_download_path}."
        )

    # Run Prediction
    pred = Prediction(wave_sounds_path, model_path)
    result = pred.prediction()

    # Returns JSON to the frontend instead of re-rendering HTML
    return JSONResponse(content={"transcription": result})

@app.get("/train")
async def train_model():
    """Trigger the training pipeline."""
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        return JSONResponse(content={"message": "Training completed"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8060)
