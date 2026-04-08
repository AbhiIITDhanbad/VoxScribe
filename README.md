<div align="center">
  <h1>🎙️ VoxScribe</h1>
  <p><strong>End-to-End Speech-to-Text Recognition MLOps Pipeline</strong></p>
</div>

---

## 📌 Project Overview
**VoxScribe** is a fully autonomous Machine Learning lifecycle system targeted at generating high-accuracy transcriptions from audio inputs. Engineered with deep modularity in mind, the project covers everything from robust cloud-based data ingestion (AWS S3) to deep neural network training, evaluation, and serving via an asynchronous FastAPI backend except the aws deployment because it is paid and costly for me like college student.

The primary dataset supported is the prestigious **LJSpeech-1.1** dataset, and the model architecture centers around a custom-built, TensorFlow-based **Speech Transformer** optimized natively without high-level abstraction bloat. 

This repository was meticulously constructed over intense development cycles to encapsulate true production standards inside an enterprise MLOps framework. VoxScribe brings together decoupled model layers, deep S3 synchronization protocols, custom logging semantics, and scalable containerization.

---

## Checkout Demo - https://drive.google.com/file/d/1b5CxrVFAzC_ofrHqNA0nSmODQLYHCnpf/view?usp=sharing
## AWS Cloud Storage Evidence - https://drive.google.com/drive/folders/1JJsR9BEPHMY6wOBhTMH7QuHer9Kr7TD8?usp=sharing

## 🧠 Neural Architecture & Deep Learning Specifications
Unlike simple API wrappers, VoxScribe implements a scratch-built Deep Neural Network using the **Keras Subclassing API**, offering granular operational control over gradient tape calculation and loss masking.

### Core Architecture Context: The Speech Transformer
- **`SpeechFeatureEmbedding`**: Extracts multi-dimensional representations (spectrogram features) directly from 1D `.wav` files into deeply dense neural embeddings.
- **Transformer Encoder**: 
  * Stacked **4 Layers** deep (`num_layers_enc = 4`).
  * Features **Multi-Head Attention** (`num_head = 2`) over a robust hidden state representation (`num_hid = 200`).
  * Feed forward dense networks expanded to `d_ff = 400`.
- **Transformer Decoder**: 
  * Built as an auto-regressive 1-layer decoder (`num_layers_dec = 1`) receiving encoder hidden states and sequential label tracking.
  * Contextual target length caps at `MAX_TARGET_LENGTH = 200`.
- **Vocabulary Mapping (`VectorizeChar`)**: Employs a specific character-level mapping corresponding to 34 output classes (English token spaces and specials).

### Optimization & Custom Schedulers
The training phase is injected with a **`CustomSchedule`** learning rate that gracefully scales through warmup and decay limits dynamically tied to epochs:
- **Optimizer**: Explicitly utilizing the `Adam` gradient descent methodology.
- **Loss Topology**: CategoricalCrossentropy featuring `label_smoothing=0.1`.
- **Tensor Handling**: Dynamic `tf.GradientTape` masking filters padding (index `0`) to prevent gradient degradation due to variable auditory lengths.

---

## 🏗️ Robust MLOps Pipeline Breakdown
VoxScribe utilizes an exceptionally strict Object-Oriented pipeline paradigm found inside `SpeechToText/components/`. The orchestration lifecycle ensures sequential data validation, artifact versioning, and environment cleanliness.

1. **`DataIngestion` Component**: 
   Connects to an AWS S3 bucket (`s3://speech-to-text-portfolio-project/`) using `S3Sync` wrappers. Safely streams and extracts `LJSpeech-1.1.tar.bz2` targeting the `download_data` cache.
2. **`DataPreprocessing` Component**:
   Conducts hard feature mapping. Organizes wave files to text token mappings via `csv` formats parsing the metadata into a 99/1 split (`TRAIN_TEST_SPLIT = 0.99`) for intensive training sets (`train.csv`) and unseen dev checks (`test.csv`). Features native audio validation and data restructuring.
3. **`ModelTrainer` Component**:
   Creates deeply optimized `tf.data.Dataset` tensors iterating natively over GPU/CPU boundaries (`batch_size=32`). Upon completing epochs, stores only core neural representations via isolated `.weights.h5` formatted checkpoints safely persisting the graph states inside `artifacts`.
4. **`ModelEvaluation` & `ModelPusher`**:
   Routinely analyzes validation loss curves. High-performing epochs trigger continuous deployments, securely transferring raw weights back into the production AWS S3 buckets automatically for inference endpoints to retrieve.

---

## 🌐 API Interaction Layer (FastAPI & Jinja2)
Once models are trained and pushed, the `fastapi_app.py` script boots a lightning-fast Uvicorn server scaling inference logic dynamically.
- Native asynchronous `UploadFile` endpoints to route audio streams functionally without heavy disk IO bottlenecks.
- S3 automated model polling guarantees the server pulls down the latest `.weights.h5` instance upon boot prediction.
- Dynamic UX rendering via `frontend.html` Jinja templating allowing users to upload `.wav` samples and securely watch the UI poll JSON endpoint payloads showing dynamic transcriptions seamlessly.

---

## 🐳 Running with Docker (Recommended)
VoxScribe uses advanced containerization parameters bridging local machine testing linearly with Enterprise deployments. The repository is completely Docker-bound.

### Prerequisites Checklist:
- <b>Docker Engine</b> configured and running locally.
- AWS <b>Access Key ID</b> and <b>Secret Access Key</b> with permissions mapped over `s3:GetObject` against the configured model bucket. 

### Step-by-Step Launch
**1. Clone into your local environment**
```bash
git clone <your-repository-url>
cd Speech-to-Text
```

**2. Isolate and System Build**
Use the integrated Dockerfile which targets `python:3.12-slim` efficiently bundling the AWS CLI bindings and pip constraints contextually.
```bash
docker build -t speech-to-text-project .
```

**3. Inject Context & Run as Daemon**
Launch the server. Ensure that the model retrieval pipelines receive proper authentication hooks through `--env` parameters enabling `boto3`/`awscli` background syncing.
```bash
docker run -d \
  -e AWS_ACCESS_KEY_ID="<INSERT_YOUR_AWS_ACCESS_KEY_ID>" \
  -e AWS_SECRET_ACCESS_KEY="<INSERT_YOUR_AWS_SECRET_ACCESS_KEY>" \
  -e AWS_DEFAULT_REGION="us-east-1" \
  -p 8060:8060 \
  speech-to-text-project
```

**4. View your Live Enterprise App**
Access the fully functional transcription UX right from your browser targeting:
```text
http://localhost:8060
```
Simply trigger an upload of an LJSpeech or generated `.wav` format, and the Deep Speech feature mapping will transcribe the sequence directly.

---

## 📜 Future CI/CD Additions
The components are intentionally organized to align closely with standard CI/CD DAGs (Airflow, GitHub Actions). In forthcoming commits, isolated step functions could replace singular orchestration routines unlocking serverless AWS SageMaker training paths.

## 📄 License Context
All deep learning implementations, scripts, configurations, and MLOps strategies reside fully under the repository's main `LICENSE`.
