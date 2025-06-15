import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from urllib.parse import urlparse
import hashlib

def clean_text(text):
    if pd.isna(text):
        return ""
    # Convert to lowercase and remove special characters
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return ' '.join(text.split())

def get_image_hash(url):
    """Extract a unique identifier from image URL by removing query parameters and timestamps"""
    if pd.isna(url):
        return ""
    try:
        # Parse URL and get path
        parsed = urlparse(url)
        path = parsed.path
        
        # Remove common image size/quality parameters from path
        path = re.sub(r'_\d+x\d+', '', path)
        path = re.sub(r'_\d+', '', path)
        
        # Create hash of the cleaned path
        return hashlib.md5(path.encode()).hexdigest()
    except:
        return ""

def calculate_image_similarity(images1, images2, max_images=5):
    """Calculate similarity between two sets of image URLs, considering only first max_images"""
    if pd.isna(images1) or pd.isna(images2):
        return 0.0
        
    # Convert string of URLs to list and take only first max_images
    if isinstance(images1, str):
        images1 = [url.strip() for url in images1.strip('[]').split(',')][:max_images]
    if isinstance(images2, str):
        images2 = [url.strip() for url in images2.strip('[]').split(',')][:max_images]
    
    # Get unique image hashes
    hashes1 = set(get_image_hash(url) for url in images1)
    hashes2 = set(get_image_hash(url) for url in images2)
    
    # Calculate Jaccard similarity
    intersection = len(hashes1.intersection(hashes2))
    union = len(hashes1.union(hashes2))
    
    return intersection / union if union > 0 else 0.0

def deduplicate_cars(input_file='cars_pred.xlsx', output_file='cars_deduplicated.xlsx', 
                    similarity_threshold=0.65, image_similarity_threshold=0.3):
    # Read the data
    df = pd.read_excel(input_file)
    print(f"Original number of entries: {len(df)}")
    
    # Clean and prepare the data
    df['description_clean'] = df['description'].apply(clean_text)
    
    # Create brand-model groups
    df['brand_model'] = df['brand'] + '_' + df['model']
    
    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=1000)
    
    # Process each brand-model group separately
    deduplicated_rows = []
    
    for brand_model, group in df.groupby('brand_model'):
        if len(group) == 1:
            deduplicated_rows.append(group.iloc[0])
            continue
            
        # Calculate TF-IDF matrix for descriptions
        tfidf_matrix = vectorizer.fit_transform(group['description_clean'])
        
        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Find similar entries
        to_keep = []
        to_remove = set()
        
        for i in range(len(group)):
            if i in to_remove:
                continue
                
            to_keep.append(i)
            
            # Find similar entries
            for j in range(i + 1, len(group)):
                if j in to_remove:
                    continue
                
                # First check image similarity
                image_similarity = calculate_image_similarity(
                    group.iloc[i]['images'], 
                    group.iloc[j]['images']
                )
                
                # If images are similar enough, consider as duplicate regardless of other parameters
                if image_similarity > image_similarity_threshold:
                    to_remove.add(j)
                    continue
                
                # If images are not similar, check text similarity
                if similarity_matrix[i, j] > similarity_threshold:
                    # If prices are significantly different, keep both
                    price_diff = abs(group.iloc[i]['median'] - group.iloc[j]['median'])
                    if price_diff > group.iloc[i]['median'] * 0.15:  # 15% price difference threshold
                        continue
                        
                    to_remove.add(j)
        
        # Add kept entries to deduplicated rows
        for idx in to_keep:
            deduplicated_rows.append(group.iloc[idx])
    
    # Create new dataframe with deduplicated entries
    deduplicated_df = pd.DataFrame(deduplicated_rows)
    
    # Drop the temporary columns
    deduplicated_df = deduplicated_df.drop(['description_clean', 'brand_model'], axis=1)
    
    # Save to new file
    deduplicated_df.to_excel(output_file, index=False)
    print(f"Deduplicated number of entries: {len(deduplicated_df)}")
    print(f"Removed {len(df) - len(deduplicated_df)} duplicate entries")

if __name__ == "__main__":
    deduplicate_cars() 