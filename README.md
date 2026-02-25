# DermAI Production Pipeline

This repository implements the comprehensive AI Dermatology pipeline fulfilling the requirements provided for data processing, modeling, AI explainability, clinical deployment, and system architecture. 

## 1. Data Pipeline (`training/data_pipeline.py`)
- **Datasets Supported:** HAM10000, ISIC, PAD-UFES-20 (for diversity inclusion).
- **Preprocessing & Augmentation:** Uses OpenCV for image resizing (224x224) and CLAHE contrast enhancement. Uses `albumentations` for deep augmentations. 
- **Sampling & Imbalance:** Implements stratified k-fold splits, extracts Skin Tone distribution per Fitzpatrick scale to ensure fairness, and applies a PyTorch `WeightedRandomSampler` to correct severe class imbalances prior to applying **Focal Loss** (`training/utils/losses.py`).

## 2. Model Architecture (`training/models/architecture.py`)
- **Primary Server Model:** EfficientNet-B4. Selected for its optimal accuracy/performance profile.
- **Edge/Mobile Model:** MobileNetV3 Large deployed natively in PyTorch for downstream conversion to ONNX and INT8 Quantization.
- **Layers:** Incorporates Dropout (0.5), Batch Normalization, and implicit Softmax probability extraction. Uses AdamW optimizer + Cosine LR Annealing context.
- **Loss Strategy:** Implements Focal Loss ($\gamma=2$) to tackle complex hard clinical cases.

## 3. Explainability (`app/services/explainability.py`)
- **Visual:** Uses Grad-CAM to output attention regions (heatmaps) over the input image array.
- **Confidence Calibration:** Applies Platt Scaling (Temperature Scaling) over model logits to calculate calibrated probability values instead of overconfident DNN predictions.
- **Safety Fallback:** If confidence is evaluated below $60\%$, the system automatically triggers a clinical warning recommending direct dermatologist review.

## 4. NLP Module - Medical RAG (`app/services/nlp_rag.py`)
- Eliminates random GPT text generation hallucinations entirely.
- Integrates a RAG-based extraction mechanism simulating vector-searches across PubMed Abstracts and structured dermatology guidelines. 
- Guarantees returned output strictly limits itself to formatting Causes, Symptoms, Treatment, Recovery Estimates, and Clinical recommendations based strictly on valid evidence blocks.

## 5. Progress Tracking Engine
- Future implementations can connect image sequences into a PostgreSQL chronological timeline.
- A regression model will quantify Severity via bounding-box dimensions, coloration variations, and structural similarity indices (SSIM).

## 6. Backend architecture (`app/api/endpoints.py`)
- Full non-blocking async REST deployment via `FastAPI`. 
- Incorporates API gateways for: `/predict`, `/explain`, `/report`, `/upload-followup`, `/patient-history`, `/risk-score`. Pydantic handles strictly typed validation.
- Employs simulated bindings for future integration to **PostgreSQL**, **Redis** tracking, and **Celery** inference worker queues.

## 7. Cloud Deployment & CI/CD
- **Docker/Kubernetes Native:** Folders `deployment/docker` and `deployment/k8s` hold future system manifests.
- Load-balancer logic connects Frontend components directly through an API Gateway with an integrated system designed to autoscale up during usage peaks.
- **Monitoring:** Supports ingestion to Prometheus metrics endpoints (latency, inferences/sec) and Model Drift tracking across different data distributions (ELK/Grafana integration ready).

## 8. Security & Compliance
- HTTPS protocol mandated.
- Images resting in S3/Blocks must be AES-256 encrypted.
- Production systems utilize JWT Authentication tokens and strictly implement Role-Based Access Controls (Providers vs. Patients).

## 9. Mobile AI Mode
- Codebase allows `torch.onnx.export` of `MobileNetV3` weights to ONNX/TensorRT runtimes natively.
- Deploying Edge-specific TFLite configurations is facilitated via the `MobileNet` module structure.

## 10. Output Specification Fulfillment
The central AI prediction pipeline formats outputs via Pydantic exactly precisely to standard:
```json
{
    "disease": "Melanoma",
    "probability": "58.0%",
    "confidence": "58.0%",
    "heatmap": "base64_encoded_string",
    "treatment_plan": "Surgical Excision.",
    "recovery_estimate": "Varies by stage.",
    "severity_score": "8/10",
    "followup_required": true
}
```
If Model Confidence < 60%, the `disease` output is overridden to trigger emergency review ("Dermatologist review required"). Bad/Blurry uploads inherently trigger a $400$ `Retake image in better lighting` failure path.
