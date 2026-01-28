import xml.etree.ElementTree as ET
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('punkt')
nltk.download('stopwords')

def extract_abstracts(xml_path, min_abstracts=10):
    """
    Extracts <AbstractText> content from a PubMed XML file.
    Returns a list of abstracts (strings).
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    abstracts = []
    for elem in root.iter():
        if elem.tag == 'AbstractText':
            text = elem.text
            if text:
                abstracts.append(text.strip())
        if len(abstracts) >= min_abstracts:
            break
    return abstracts

def segment_sentences(abstracts):
    """
    Segments each abstract into sentences using nltk.
    Returns a list of sentences.
    """
    sentences = []
    for abstract in abstracts:
        sents = nltk.sent_tokenize(abstract)
        sentences.extend(sents)
    return sentences

def clean_sentence(sentence):
    """
    Clean sentence by removing punctuation and weird symbols.
    """
    # Remove URLs
    sentence = re.sub(r'https?://\S+|www\.\S+', '', sentence)
    
    # Remove email addresses
    sentence = re.sub(r'\S+@\S+', '', sentence)
    
    # Remove special characters and punctuation (keep letters, numbers, and basic punctuation for sentence splitting)
    sentence = re.sub(r'[^\w\s\.\?\!]', ' ', sentence)
    
    # Remove extra whitespace
    sentence = re.sub(r'\s+', ' ', sentence)
    
    return sentence.strip()

def preprocess_for_topic_modeling(sentences):
    """
    Preprocess sentences for topic modeling:
    1. Clean sentences (remove punctuation, weird symbols)
    2. Lowercase
    3. Remove stop words
    4. Apply stemming
    
    Returns a list of preprocessed sentences.
    """
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    
    preprocessed_sentences = []
    
    for sentence in sentences:
        # Clean the sentence first
        cleaned_sentence = clean_sentence(sentence)
        
        # Tokenize
        tokens = nltk.word_tokenize(cleaned_sentence)
        
        # Lowercase, remove stop words, and stem
        processed_tokens = []
        for token in tokens:
            # Skip punctuation tokens that survived tokenization
            if token in ['.', ',', ';', ':', '!', '?', "'", '"', '(', ')', '[', ']', '{', '}']:
                continue
                
            # Lowercase
            token_lower = token.lower()
            
            # Remove non-alphabetic tokens and stop words
            if token_lower.isalpha() and token_lower not in stop_words and len(token_lower) > 1:
                # Apply stemming
                stemmed_token = stemmer.stem(token_lower)
                processed_tokens.append(stemmed_token)
        
        # Reconstruct the sentence with processed tokens
        if processed_tokens:  # Only add if there are tokens left
            processed_sentence = ' '.join(processed_tokens)
            preprocessed_sentences.append(processed_sentence)
    
    return preprocessed_sentences

def preprocess_for_general_use(sentences):
    """
    Preprocess sentences for general use (morphosyntax, POS tagging, etc.):
    1. Clean sentences (remove weird symbols but keep basic punctuation)
    2. Normalize whitespace
    
    Returns a list of cleaned sentences.
    """
    cleaned_sentences = []
    
    for sentence in sentences:
        # Clean the sentence but keep basic punctuation for linguistic analysis
        cleaned = clean_sentence(sentence)
        
        # Remove trailing punctuation that might be left
        cleaned = re.sub(r'^[\.\?\!\,\;\:]+|[\.\?\!\,\;\:]+$', '', cleaned)
        
        if cleaned:  # Only add if not empty after cleaning
            cleaned_sentences.append(cleaned)
    
    return cleaned_sentences

def save_sentences(sentences, out_path):
    """Save sentences to a file, one per line."""
    with open(out_path, 'w', encoding='utf-8') as f:
        for sent in sentences:
            f.write(sent.strip() + '\n')

def preprocess_pubmed(xml_path, out_path_base, min_abstracts=10):
    """
    Main preprocessing function that:
    1. Extracts abstracts from PubMed XML
    2. Segments into sentences
    3. Cleans sentences for general use and saves to sentences.txt
    4. Preprocesses for topic modeling and saves to sentences_tm.txt
    
    Args:
        xml_path: Path to PubMed XML file
        out_path_base: Base path for output files (without extension)
        min_abstracts: Minimum number of abstracts to extract
    """
    # Extract and segment sentences
    abstracts = extract_abstracts(xml_path, min_abstracts)
    sentences = segment_sentences(abstracts)
    
    # Clean sentences for general use (morphosyntax, POS, etc.)
    cleaned_sentences = preprocess_for_general_use(sentences)
    
    # Save cleaned sentences
    orig_path = out_path_base + '.txt'
    save_sentences(cleaned_sentences, orig_path)
    
    # Preprocess for topic modeling and save
    tm_sentences = preprocess_for_topic_modeling(cleaned_sentences)
    tm_path = out_path_base + '_tm.txt'
    save_sentences(tm_sentences, tm_path)
    
    print(f"Extracted {len(sentences)} sentences from {len(abstracts)} abstracts.")
    print(f"After cleaning: {len(cleaned_sentences)} sentences")
    print(f"Saved cleaned sentences to: {orig_path}")
    print(f"Saved preprocessed sentences for topic modeling to: {tm_path}")
    
    # Show samples
    if cleaned_sentences:
        print(f"\nSample cleaned sentence: {cleaned_sentences[0][:100]}...")
    if tm_sentences:
        print(f"Sample topic modeling sentence: {tm_sentences[0][:100]}...")

if __name__ == "__main__":
    xml_path = os.path.join('data', 'pubmed.xml')
    out_path_base = os.path.join('data', 'sentences')
    os.makedirs('data', exist_ok=True)
    preprocess_pubmed(xml_path, out_path_base, min_abstracts=10)