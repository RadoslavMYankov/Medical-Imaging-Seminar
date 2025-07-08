# Medical-Imaging-Seminar
Repository to evaluate a multimodal foundation model on medical data (chest X-Ray and brain MRI)

## Overview

This repository contains two Jupyter notebooks that evaluate the **LLaVA-Med v1.5 Mistral 7B** multimodal foundation model on different medical imaging tasks using two distinct datasets:

## Notebooks

### 1. `chest_xray.ipynb`
**Dataset:** Chest X-ray images  
**Tasks:**
- **Classification:** Binary classification of X-ray images as "healthy" or "unhealthy"
- **Localization:** Object detection and grounding of specific diseases/abnormalities by generating bounding box coordinates

**Key Features:**
- Loads chest X-ray images and their corresponding annotations
- Performs binary health status classification
- Attempts to locate specific diseases mentioned in the annotations using natural language prompts
- Outputs results to CSV files for further analysis

### 2. `brain_mri.ipynb`
**Dataset:** Brain MRI scans  
**Tasks:**
- **Captioning:** Generates descriptive captions for brain MRI images, highlighting the most important medical findings
- **Localization:** Identifies and localizes abnormal regions in brain scans by providing bounding box coordinates

**Key Features:**
- Processes brain MRI images for medical description generation
- Creates detailed captions describing medical findings
- Performs abnormal region detection and grounding
- Saves predictions and descriptions to CSV files

## Model
Both notebooks utilize the **Eren-Senoglu/llava-med-v1.5-mistral-7b-hf** model, a medical-focused vision-language model capable of understanding and analyzing medical images through natural language instructions.

## Output
The notebooks generate CSV files containing:
- Image filenames
- Model predictions/descriptions
- Bounding box coordinates (for localization tasks)