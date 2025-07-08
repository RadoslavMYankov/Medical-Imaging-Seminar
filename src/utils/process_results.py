import pandas as pd
import json
from src.data.scripts.evaluate_metrics import accuracy_score, f1_score

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
    output.to_csv("/home/useradd/seminar/Mecial-Imaging-Seminar/results/xray_classifications.csv", index=False)
    print("Output saved to xray_classifications.csv")

def get_metrics(results):
    gt = results['label'].tolist()
    pred = results['prediction'].tolist()

    accuracy = accuracy_score(gt, pred)
    f1 = f1_score(gt, pred, pos_label='unhealthy')

    # Print results
    print(f"Ground Truth: {gt}")
    print(f"Predictions: {pred}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"F1 Score: {f1:.3f}")
if __name__ == "__main__":
    with open("/home/useradd/seminar/Mecial-Imaging-Seminar/src/data/chest_xrays/annotations_len_50.json", "r") as f:
        data = json.load(f)
    predictions = pd.read_csv("/home/useradd/seminar/Mecial-Imaging-Seminar/results/xray_predictions.csv")
    
    #output = get_classifications_xray(data, predictions)
    results = pd.read_csv("/home/useradd/seminar/Mecial-Imaging-Seminar/results/xray_classifications.csv")
    get_metrics(results)

    