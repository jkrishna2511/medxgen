# medxgen
Exploring deep learning techniques to model the relationship between medical images and clinical text.

# MedExGen: Medical X-Ray Report Generation Pipeline

**MedExGen** is a deep learning system that generates medical reports from chest X-ray images using a ResNet50-GPT2 encoder-decoder architecture.

## How to Run

Open and run the notebook `medexgen-pipeline.ipynb` **cell by cell**. It handles everything from dataset download to training, evaluation, and visualization.

---

## Project Structure

### `./IUXRAY_DATA_PIPELINE/`  
Stores the IU X-Ray dataset  
- `images/`: X-ray PNG image files  
- `reports/`: XML clinical report files  
- `temp/`: Temporary directory for downloads and extraction  

### `./IUXRAY_OUTPUT_PIPELINE/`  
Stores all model outputs and visualizations  
- `checkpoints/`: Model checkpoints saved during training  
- `plots/`: Training curves, prediction samples, and other visualizations  
- `logs/`: Evaluation metrics and prediction outputs in JSON format  
- `models/`: Final model files (if exported separately)  

---

## Running the Pipeline

The notebook performs the following:

- Downloads and organizes the IU X-Ray dataset  
- Parses XML reports and links them to X-ray images  
- Performs data exploration and statistical analysis  
- Preprocesses images and tokenizes text  
- Trains a ResNet50-GPT2 model with gradient accumulation  
- Evaluates performance using BLEU, ROUGE, and medical F1 metrics  
- Visualizes training progress and sample outputs  

---

## Key Components

- **Data Loading**: Automatically downloads and extracts the dataset  
- **Preprocessing**: Normalizes images and tokenizes reports using GPT-2 tokenizer with custom medical tokens  
- **Architecture**: ResNet50 as image encoder, GPT2 as decoder, with a linear projection layer  
- **Training**: AdamW optimizer, mixed learning rates, early stopping, and learning rate scheduling  
- **Evaluation**: BLEU (1–4), ROUGE (1, 2, L), and condition-level detection scores (F1) for terms like "effusion", "edema", etc.

---

## Model Outputs

- Checkpoints saved at: `./IUXRAY_OUTPUT_PIPELINE/checkpoints/`  
- Visualizations and plots at: `./IUXRAY_OUTPUT_PIPELINE/plots/`  
- Logs and evaluation results at: `./IUXRAY_OUTPUT_PIPELINE/logs/`  

---

## Visualizing Results

- `visualize_model_predictions()` loads the best model and generates sample predictions from test images  
- `visualize_training_summary()` displays loss curves and validation metrics across epochs  

---

## Note on Epoch 37

A spike in metrics at epoch 37 was caused by an accidental change in the validation script. The evaluation method was temporarily altered, which led to inflated scores. This was later reverted to restore consistent evaluation across all epochs.
