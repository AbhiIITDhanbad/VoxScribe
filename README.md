<div align="center">
  <h1>🎙️ VoxScribe</h1>
  <p><strong>Production-Grade End-to-End Speech-to-Text MLOps Pipeline</strong></p>
  <p>
    <img src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white" alt="Python 3.12"/>
    <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white" alt="TensorFlow"/>
    <img src="https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white" alt="FastAPI"/>
    <img src="https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white" alt="Docker"/>
    <img src="https://img.shields.io/badge/AWS_S3-Integrated-232F3E?logo=amazonaws&logoColor=white" alt="AWS S3"/>
  </p>
</div>

---

## 📌 Overview

**VoxScribe** is a fully modular, production-ready Machine Learning system that transcribes spoken audio into text. It implements the complete ML lifecycle — from cloud-based data ingestion, to deep neural network training and evaluation, to model versioning and serving — all wrapped inside a clean MLOps framework with artifact tracking, structured logging, and containerized deployment.

The project is trained on the [**LJSpeech-1.1**](https://keithito.com/LJ-Speech-Dataset/) dataset and uses a custom-built **Speech Transformer** architecture implemented from scratch with the TensorFlow/Keras Subclassing API. Inference is served through a modern glassmorphism-styled web interface backed by an async **FastAPI** server.

> **Note:** Cloud deployment on AWS was intentionally skipped to avoid costs, but the pipeline is fully wired for it — models are synced to/from S3, and the Docker image is production-ready.

---

## 🎬 Demo

| Resource | Link |
|---|---|
| 🎥 **Video Demo** | [Watch on Google Drive](https://drive.google.com/file/d/1b5CxrVFAzC_ofrHqNA0nSmODQLYHCnpf/view?usp=sharing) |
| ☁️ **AWS S3 Evidence** | [View on Google Drive](https://drive.google.com/drive/folders/1JJsR9BEPHMY6wOBhTMH7QuHer9Kr7TD8?usp=sharing) |

---

## 🧠 Model Architecture — Speech Transformer

VoxScribe does **not** wrap a third-party API. The entire neural network is built from scratch using `keras.Model` subclassing, providing full control over the training loop, gradient computation, and loss masking.

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Input (.wav audio)                        │
│                          │                                  │
│                 ┌────────▼────────┐                         │
│                 │ SpeechFeature   │  3× Conv1D layers       │
│                 │ Embedding       │  (kernel=11, stride=2)  │
│                 └────────┬────────┘                         │
│                          │                                  │
│              ┌───────────▼───────────┐                      │
│              │  Transformer Encoder  │  4 stacked layers    │
│              │  (Multi-Head Attn +   │  num_heads=2         │
│              │   Feed-Forward)       │  d_model=200         │
│              └───────────┬───────────┘  d_ff=400            │
│                          │                                  │
│              ┌───────────▼───────────┐                      │
│              │  Transformer Decoder  │  1 layer             │
│              │  (Causal Masked Attn  │  Auto-regressive     │
│              │   + Cross-Attention)  │                      │
│              └───────────┬───────────┘                      │
│                          │                                  │
│                 ┌────────▼────────┐                         │
│                 │ Dense Classifier│  34 output classes      │
│                 └────────┬────────┘                         │
│                          │                                  │
│                 Character-level text                        │
└─────────────────────────────────────────────────────────────┘
```

### Key Technical Details

| Component | Specification |
|---|---|
| **Audio Preprocessing** | STFT (frame_length=200, frame_step=80, fft_length=256) → power-law compression → mean/std normalization → pad to 2754 frames |
| **Encoder** | 4-layer Transformer with Multi-Head Attention (2 heads), LayerNorm, Dropout(0.1), FFN(200→400→200) |
| **Decoder** | 1-layer auto-regressive Transformer with causal masking, self-attention + cross-attention over encoder states |
| **Vocabulary** | 34-class character-level tokenizer (`VectorizeChar`): a-z, space, period, comma, question mark, and special tokens (`<`, `>`, `-`, `#`) |
| **Max Target Length** | 200 tokens |
| **Loss Function** | CategoricalCrossentropy with `label_smoothing=0.1`, masked to ignore padding (index 0) |
| **Optimizer** | Adam with `CustomSchedule` — warm-up over 15 epochs (1e-5 → 1e-3), then linear decay over 40 epochs back to 1e-5 |
| **Training** | Custom `train_step` with `tf.GradientTape` for explicit gradient masking of padded positions |
| **Inference** | Greedy auto-regressive decoding via `model.generate()` |

---

## 🏗️ MLOps Pipeline

The training lifecycle is orchestrated by `TrainingPipeline`, which executes five decoupled components in strict sequence. Every component produces typed **artifact dataclasses** that are consumed by the next stage, ensuring clean interfaces and reproducibility.

```
DataIngestion ──▶ DataPreprocessing ──▶ ModelTrainer ──▶ ModelEvaluation ──▶ ModelPusher
     │                   │                   │                  │                │
  S3 → local         metadata.csv      tf.data.Dataset     Compare with     Sync best
  tar.bz2 extract    train/test split   model.fit()       S3 production    to S3 bucket
                     (99/1 split)       .weights.h5        model loss
```

### Component Details

| Stage | Module | Responsibility |
|---|---|---|
| **Data Ingestion** | `components/data_ingestion.py` | Downloads `LJSpeech-1.1.tar.bz2` from AWS S3 bucket via `aws s3 cp`, extracts to `data/LJSpeech-1.1/`. Skips download if data already exists locally. |
| **Data Preprocessing** | `components/data_preprocessing.py` | Parses `metadata.csv` to build `{id → text}` mappings, globs all `.wav` files, creates structured `train.csv` / `test.csv` splits (99/1 ratio). |
| **Model Trainer** | `components/model_trainer.py` | Creates `tf.data.Dataset` pipelines (batch=32 train, batch=8 val) with caching and prefetching. Builds the Transformer, compiles, trains, and saves `.weights.h5` checkpoints. |
| **Model Evaluation** | `components/model_evaluation.py` | Pulls the current production model from S3, evaluates both models on validation data, and determines whether the newly trained model should replace production. |
| **Model Pusher** | `components/model_pusher.py` | If the new model's loss is lower than S3 production, syncs the trained weights to the S3 model bucket via `aws s3 sync`. |

### Artifact Versioning

All artifacts are stored under a timestamped directory:
```
artifacts/
  └── MM_DD_YYYY_HH_MM_SS/
        ├── data_preprocessing_artifacts/
        │     ├── metadata/wavs.csv
        │     ├── train/train.csv
        │     └── test/test.csv
        ├── model_trainer_artifact/
        │     └── saved_model/model.weights.h5
        ├── model_evaluation_artifact/
        │     └── s3_model/  (downloaded production weights)
        └── model_pusher/
```

---

## 🌐 API & Web Interface

### FastAPI Backend (`fastapi_app.py`)

| Endpoint | Method | Description |
|---|---|---|
| `/` | `GET` | Serves the glassmorphism-styled frontend (`frontend.html`) |
| `/predict` | `POST` | Accepts audio file upload, syncs latest model from S3, runs inference, returns JSON `{"transcription": "..."}` |
| `/train` | `GET` | Triggers the full training pipeline end-to-end |

### Frontend

A single-page interface built with vanilla HTML/CSS/JS featuring:
- **Glassmorphism** dark theme with gradient accents
- **Drag-and-drop** audio upload with file validation
- **Async fetch** to `/predict` — no page reloads
- Real-time loading spinner and error states
- One-click **copy to clipboard** for transcription results

---

## 📂 Project Structure

```
VoxScribe/
├── SpeechToText/                    # Core Python package
│   ├── cloud_storage/
│   │   └── s3_operations.py         # S3Sync wrapper (upload/download/sync)
│   ├── components/
│   │   ├── data_ingestion.py        # S3 data download & extraction
│   │   ├── data_preprocessing.py    # Metadata parsing, train/test split
│   │   ├── model_trainer.py         # Dataset creation, model training
│   │   ├── model_evaluation.py      # S3 model comparison & validation
│   │   └── model_pusher.py          # Conditional model deployment to S3
│   ├── configuration/               # (Reserved for future config managers)
│   ├── constants/
│   │   └── __init__.py              # All project-wide constants & hyperparams
│   ├── entity/
│   │   ├── artifact_entity.py       # Typed artifact dataclasses
│   │   ├── config_entity.py         # Pipeline stage configuration dataclasses
│   │   └── model_entity.py          # CreateTensors — tf.data pipeline builder
│   ├── exceptions/
│   │   └── __init__.py              # STTException with traceback details
│   ├── logger/
│   │   └── __init__.py              # Timestamped file-based logging (logs/)
│   ├── models/
│   │   ├── data_utils.py            # VectorizeChar tokenizer & get_data helper
│   │   ├── model.py                 # Transformer model (train/test/generate)
│   │   └── model_utils.py           # Encoder, Decoder, Embeddings, LR Schedule
│   ├── pipeline/
│   │   ├── training_pipeline.py     # End-to-end training orchestrator
│   │   └── prediction_pipeline.py   # Single-audio inference pipeline
│   └── utils/
│       └── prediction_utils.py      # Audio-to-spectrogram conversion for inference
│
├── template/
│   └── frontend.html                # Modern glassmorphism web UI (Jinja2)
├── fastapi_app.py                   # FastAPI application entry point
├── app.py                           # Legacy Flask application (deprecated)
├── demo.py                          # Quick-test script for local predictions
├── template.py                      # Project scaffolding / boilerplate generator
├── Dockerfile                       # Production container (python:3.12-slim)
├── .dockerignore                    # Docker build exclusions
├── requirements.txt                 # Python dependencies
├── setup.py                         # Package installation (pip install -e .)
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- **Docker Engine** installed and running, OR **Python 3.12+** with pip
- AWS **Access Key ID** and **Secret Access Key** with `s3:GetObject` / `s3:PutObject` permissions on the configured bucket (`speech-to-text-portfolio-project`)

---

### 🐳 Option 1: Run with Docker (Recommended)

**1. Clone the repository**
```bash
git clone https://github.com/AbhiIITDhanbad/VoxScribe.git
cd VoxScribe
```

**2. Build the Docker image**
```bash
docker build -t voxscribe .
```

**3. Run the container**
```bash
docker run -d \
  -e AWS_ACCESS_KEY_ID="<YOUR_AWS_ACCESS_KEY_ID>" \
  -e AWS_SECRET_ACCESS_KEY="<YOUR_AWS_SECRET_ACCESS_KEY>" \
  -e AWS_DEFAULT_REGION="us-east-1" \
  -p 8060:8060 \
  voxscribe
```

**4. Open in browser**
```
http://localhost:8060
```

Upload a `.wav` file and get your transcription!

---

### 💻 Option 2: Run Locally

**1. Clone and install dependencies**
```bash
git clone https://github.com/AbhiIITDhanbad/VoxScribe.git
cd VoxScribe
pip install -r requirements.txt
```

**2. Configure AWS credentials**
```bash
export AWS_ACCESS_KEY_ID="<YOUR_KEY>"
export AWS_SECRET_ACCESS_KEY="<YOUR_SECRET>"
export AWS_DEFAULT_REGION="us-east-1"
```

**3. Start the server**
```bash
python fastapi_app.py
```

**4. (Optional) Trigger training**
```
GET http://localhost:8060/train
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Deep Learning** | TensorFlow / Keras (Subclassing API) |
| **API Framework** | FastAPI + Uvicorn |
| **Frontend** | Vanilla HTML/CSS/JS, Jinja2 templating |
| **Cloud Storage** | AWS S3 (via AWS CLI) |
| **Containerization** | Docker (python:3.12-slim base) |
| **Audio Processing** | TensorFlow STFT, pydub (MP3→WAV) |
| **Logging** | Python `logging` module → timestamped files |
| **Package Management** | setuptools, pip |
| **Dataset** | LJSpeech-1.1 (~13,100 audio clips, ~24 hours) |

---

## 🗺️ Future Roadmap

- [ ] CI/CD pipeline with GitHub Actions for automated training & deployment
- [ ] Integration with AWS SageMaker for scalable cloud training
- [ ] Airflow DAG orchestration for scheduled retraining
- [ ] Beam search decoding for improved transcription quality
- [ ] Support for additional datasets and multilingual models
- [ ] Real-time microphone streaming via WebSocket

---

## 👤 Author

**Abhirup Sarkar**

---

## 📄 License

This project and all its implementations, scripts, and configurations are available under the repository's [LICENSE](./LICENSE).
