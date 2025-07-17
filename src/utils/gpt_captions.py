import pandas as pd
import openai
import os
from typing import Optional

# Set up OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def rank_caption_with_gpt(caption: str, label: str) -> Optional[int]:
    """
    Ask GPT-4o-mini to rank how well the caption matches the label.
    
    Returns:
        int: Rating from 1-5 where:
        1 - completely wrong
        3 - moderately ok  
        5 - meaning is exactly the same
    """
    prompt = f"""
    Please rate how well this medical image caption matches the given label on a scale of 1-5:
    
    Caption: "{caption}"
    Label: "{label}"
    
    Rating scale:
    1 - completely wrong
    2 - mostly wrong
    3 - moderately ok
    4 - mostly correct
    5 - meaning is exactly the same
    
    Respond with only the number (1-5).
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert medical imaging analyst. Rate caption accuracy objectively."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=10,
            temperature=0.1
        )
        
        rating = int(response.choices[0].message.content.strip())
        if 1 <= rating <= 5:
            return rating
        else:
            print(f"Invalid rating received: {rating}. Expected 1-5.")
            return None
        
    except Exception as e:
        print(f"Error calling GPT API: {e}")
        return None

results = pd.read_csv("/home/useradd/seminar/Mecial-Imaging-Seminar/results/mri_captions_v2.csv")
ratings = []

for index, row in results.iterrows():
    caption = row['caption']
    label = row['label']
    
    # Ask gpt-4o-mini to rank the caption out of 5
    rating = rank_caption_with_gpt(caption, label)
    ratings.append(rating)
    
    print(f"Row {index}: Rating = {rating}")

# Add ratings column to the original dataframe
results['gpt_rating'] = ratings

# Save back to the original CSV file
results.to_csv("/home/useradd/seminar/Mecial-Imaging-Seminar/results/mri_captions_v2.csv", index=False)

print("GPT ratings appended to original CSV file successfully!")