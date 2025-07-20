import pandas as pd
import json
import openai
import os
from typing import Optional

# Set up OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def evaluate_with_gpt(diagnosis: str, label: str) -> Optional[int]:
    """
    Given a medical image diagnosis and a label, use GPT-4o-mini to rank the accuracy of the diagnosis.
    """
    prompt = f"""
    Given the diagnosis and label, please determine if the diagnosis is correct or incorrect.
    
    Diagnosis: "{diagnosis}"
    Label: "{label}"

    Respond only with the label 'correct' or 'incorrect'.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a medical expert. Determine the accuracy of the diagnosis."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=10,
            temperature=0.1
        )
        
        rating = response.choices[0].message.content.strip().lower()
        if rating in ['correct', 'incorrect']:
            return rating
        else:
            print(f"Invalid rating received: {rating}. Expected 'correct' or 'incorrect'.")
            return None
        
    except Exception as e:
        print(f"Error calling GPT API: {e}")
        return None

results = pd.read_csv("/home/useradd/seminar/Medical-Imaging-Seminar/results/mri_predictions_diagnosis.csv")
ratings = []

with open("/home/useradd/seminar/Medical-Imaging-Seminar/src/data/nova_brain/annotations.json", 'r') as f:
    labels = json.load(f)

# Add labels to results DataFrame
for case_id in list(labels.keys()):
    gt_label = labels[case_id]['final_diagnosis']
    #create a label column in results DataFrame
    results.loc[results['case'] == case_id, 'label'] = gt_label

for index, row in results.iterrows():
    prediction = row['prediction']
    label = row['label']

    # Ask gpt-4o-mini to rank the prediction
    rating = evaluate_with_gpt(prediction, label)
    ratings.append(rating)
    
    print(f"Row {index}: Accuracy = {rating}")

# Add ratings column to the original dataframe
results['gpt_rating'] = ratings

# Save back to the original CSV file
results.to_csv("/home/useradd/seminar/Medical-Imaging-Seminar/results/mri_predictions_diagnosis.csv", index=False)
