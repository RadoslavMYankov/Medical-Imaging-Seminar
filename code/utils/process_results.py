import pandas as pd
import json
from sklearn.metrics import accuracy_score, f1_score
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading NLTK punkt_tab tokenizer...")
    nltk.download('punkt_tab')


def get_classifications_xray(data, predictions):
    output_data = []
    # strip .png from the filenames
    predictions['image'] = predictions['image'].str.replace('.png', '', regex=False)
    
    for key in data.keys():
        # Find the prediction for this image
        pred_row = predictions[predictions['image'] == key]
        
        if not pred_row.empty:
            prediction_text = pred_row['prediction'].iloc[0]
            # Extract classification from the prediction text
            if "unhealthy" in prediction_text.lower():
                prediction = "unhealthy"
            else:
                prediction = "healthy"
        else:
            prediction = "unknown"  # Handle case where no prediction is found
        
        output_data.append({
            "id": key,
            "prediction": prediction,
            "label": data[key]['status']
        })
    
    output = pd.DataFrame(output_data)
    #print(output.head())
    output.to_csv("C:\\Users\\evaka\\OneDrive\\Desktop\\rado\\Medical-Imaging-Seminar\\results\\xray_classifications.csv", index=False)
    print("Output saved to xray_classifications.csv")

def get_metrics_xray_classification(results):
    gt = results['label'].tolist()
    pred = results['prediction'].tolist()

    accuracy = accuracy_score(gt, pred)
    f1 = f1_score(gt, pred, pos_label='unhealthy')

    # Print results
    #print(f"Ground Truth: {gt}")
    #print(f"Predictions: {pred}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"F1 Score: {f1:.3f}")

def get_captions_mri(data, predictions):
    output_data = []    
    for key in data.keys():
        # Find the prediction for this image
        image_findings = data[key]['image_findings']
        for img in image_findings:
            pred_row = predictions[predictions['image'] == img]
            
            if not pred_row.empty:
                prediction_text = pred_row['prediction'].iloc[0]
            else:
                prediction_text = "No prediction available"
            
            output_data.append({
                "image": img,
                "caption": prediction_text,
                "label": image_findings[img]['caption']
            })
        
    output = pd.DataFrame(output_data)
    print(output.head())
    output.to_csv("C:\\Users\\evaka\\OneDrive\\Desktop\\rado\\Medical-Imaging-Seminar\\results\\mri_captions.csv", index=False)
    print("Output saved to mri_captions.csv")

def calculate_bleu_scores(captions_df):
    bleu_scores = []
    
    for index, row in captions_df.iterrows():
        try:
            # Handle NaN or None values
            reference_text = str(row['label']) if pd.notna(row['label']) else ""
            candidate_text = str(row['caption']) if pd.notna(row['caption']) else ""
            
            # Skip empty texts
            if not reference_text.strip() or not candidate_text.strip():
                bleu_scores.append({
                    "image": row['image'],
                    "bleu1": 0.0,
                    "bleu2": 0.0,
                    "bleu3": 0.0,
                    "bleu4": 0.0
                })
                continue
            
            reference = word_tokenize(reference_text.lower())
            candidate = word_tokenize(candidate_text.lower())
            
            # Skip if either tokenization resulted in empty list
            if not reference or not candidate:
                bleu_scores.append({
                    "image": row['image'],
                    "bleu1": 0.0,
                    "bleu2": 0.0,
                    "bleu3": 0.0,
                    "bleu4": 0.0
                })
                continue
            
            # Calculate BLEU-1,2,3,4 scores
            bleu1 = sentence_bleu([reference], candidate, weights=(1, 0, 0, 0))
            bleu2 = sentence_bleu([reference], candidate, weights=(0.5, 0.5, 0, 0))
            bleu3 = sentence_bleu([reference], candidate, weights=(0.33, 0.33, 0.33, 0))
            bleu4 = sentence_bleu([reference], candidate, weights=(0.25, 0.25, 0.25, 0.25))
            
            bleu_scores.append({
                "image": row['image'],
                "bleu1": bleu1,
                "bleu2": bleu2,
                "bleu3": bleu3,
                "bleu4": bleu4
            })
            
        except Exception as e:
            print(f"Error processing image {row['image']}: {e}")
            # Add zero scores for failed cases
            bleu_scores.append({
                "image": row['image'],
                "bleu1": 0.0,
                "bleu2": 0.0,
                "bleu3": 0.0,
                "bleu4": 0.0
            })
    
    return pd.DataFrame(bleu_scores)

if __name__ == "__main__":
    '''with open("C:\\Users\\evaka\\OneDrive\\Desktop\\rado\\Medical-Imaging-Seminar\\code\\data\\chest_xrays\\annotations_len_50.json", "r") as f:
        data = json.load(f)
    predictions = pd.read_csv("C:\\Users\\evaka\\OneDrive\\Desktop\\rado\\Medical-Imaging-Seminar\\results\\xray_predictions.csv")

    get_classifications_xray(data, predictions)
    xray_results=pd.read_csv("C:\\Users\\evaka\\OneDrive\\Desktop\\rado\\Medical-Imaging-Seminar\\results\\xray_classifications.csv")
    get_metrics_xray_classification(xray_results)'''

    with open("C:\\Users\\evaka\\OneDrive\\Desktop\\rado\\Medical-Imaging-Seminar\\code\\data\\nova_brain\\annotations.json", "r") as f:
        data = json.load(f)

    predictions = pd.read_csv("C:\\Users\\evaka\\OneDrive\\Desktop\\rado\\Medical-Imaging-Seminar\\results\\mri_descriptions.csv")
    get_captions_mri(data, predictions)

    mri_results = pd.read_csv("C:\\Users\\evaka\\OneDrive\\Desktop\\rado\\Medical-Imaging-Seminar\\results\\mri_captions.csv")
    bleu_scores = calculate_bleu_scores(mri_results)
    bleu_scores.to_csv("C:\\Users\\evaka\\OneDrive\\Desktop\\rado\\Medical-Imaging-Seminar\\results\\mri_bleu_scores.csv", index=False)
    print("BLEU scores saved to mri_bleu_scores.csv")
