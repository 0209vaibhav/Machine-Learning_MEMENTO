import nltk

def download_nltk_data():
    """Download required NLTK data"""
    resources = ['punkt', 'stopwords', 'wordnet']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            print(f"Downloading {resource}...")
            nltk.download(resource)
            print(f"Downloaded {resource}")

if __name__ == "__main__":
    download_nltk_data() 