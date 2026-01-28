import spacy
from collections import Counter
import nltk
nltk.download('punkt')

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# NER Tag descriptions
NER_DESCRIPTIONS = {
    'O': 'Outside of named entity',
    'PERSON': 'People, including fictional',
    'NORP': 'Nationalities or religious or political groups',
    'FAC': 'Buildings, airports, highways, bridges, etc.',
    'ORG': 'Companies, agencies, institutions, etc.',
    'GPE': 'Countries, cities, states',
    'LOC': 'Non-GPE locations, mountain ranges, bodies of water',
    'PRODUCT': 'Objects, vehicles, foods, etc. (not services)',
    'EVENT': 'Named hurricanes, battles, wars, sports events, etc.',
    'WORK_OF_ART': 'Titles of books, songs, etc.',
    'LAW': 'Named documents made into laws',
    'LANGUAGE': 'Any named language',
    'DATE': 'Absolute or relative dates or periods',
    'TIME': 'Times smaller than a day',
    'PERCENT': 'Percentage, including "%"',
    'MONEY': 'Monetary values, including unit',
    'QUANTITY': 'Measurements, as of weight or distance',
    'ORDINAL': 'First, second, etc.',
    'CARDINAL': 'Numerals that do not fall under another type'
}

def get_word_ner_pairs(sentences):
    """
    Get (word, NER tag) pairs for all sentences.
    
    Args:
        sentences: List of sentences
    
    Returns:
        List of tuples (word, NER_tag)
    """
    word_ner_pairs = []
    for sent in sentences:
        doc = nlp(sent)
        for token in doc:
            word_ner_pairs.append((token.text, token.ent_type_ if token.ent_type_ else 'O'))
    return word_ner_pairs

def get_ner_tags_only(sentences):
    """
    Get only NER tags for all sentences.
    
    Args:
        sentences: List of sentences
    
    Returns:
        List of NER tags
    """
    ner_tags = []
    for sent in sentences:
        doc = nlp(sent)
        for token in doc:
            ner_tags.append(token.ent_type_ if token.ent_type_ else 'O')
    return ner_tags

def get_ner_frequencies(sentences):
    """
    Get NER tag frequencies.
    
    Args:
        sentences: List of sentences
    
    Returns:
        Counter object with NER tag frequencies
    """
    ner_tags = get_ner_tags_only(sentences)
    return Counter(ner_tags)

def get_ner_bigram_frequencies(sentences):
    """
    Get NER bigram frequencies.
    
    Args:
        sentences: List of sentences
    
    Returns:
        Counter object with NER bigram frequencies
    """
    ner_tags = get_ner_tags_only(sentences)
    bigrams = list(nltk.bigrams(ner_tags))
    return Counter(bigrams)

def get_ner_statistics(sentences):
    """
    Get comprehensive NER statistics.
    
    Args:
        sentences: List of sentences
    
    Returns:
        Dictionary with all NER statistics
    """
    word_ner_pairs = get_word_ner_pairs(sentences)
    ner_tags = get_ner_tags_only(sentences)
    ner_freq = get_ner_frequencies(sentences)
    ner_bigram_freq = get_ner_bigram_frequencies(sentences)
    
    return {
        'word_ner_pairs': word_ner_pairs,
        'ner_tags': ner_tags,
        'ner_frequencies': ner_freq,
        'ner_bigram_frequencies': ner_bigram_freq,
        'total_ner_tags': len(ner_tags),
        'unique_ner_tags': len(ner_freq)
    }

def load_sentences(path):
    """Load sentences from file"""
    with open(path, encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

if __name__ == "__main__":
    # Test the functions
    sentences = load_sentences('data/sentences.txt')
    
    # Get comprehensive NER statistics
    ner_stats = get_ner_statistics(sentences)
    
    print("NER Statistics:")
    print(f"Total tokens: {ner_stats['total_ner_tags']}")
    print(f"Unique NER tags: {ner_stats['unique_ner_tags']}")
    
    print("\nNER Tag Frequencies:")
    for tag, freq in ner_stats['ner_frequencies'].most_common():
        desc = NER_DESCRIPTIONS.get(tag, "Unknown")
        print(f"{tag}: {freq} ({desc})")
    
    print("\nTop 10 NER Bigrams:")
    for bigram, freq in ner_stats['ner_bigram_frequencies'].most_common(10):
        tag1, tag2 = bigram
        desc1 = NER_DESCRIPTIONS.get(tag1, "Unknown")
        desc2 = NER_DESCRIPTIONS.get(tag2, "Unknown")
        print(f"({tag1} → {tag2}): {freq} ({desc1} → {desc2})")
    
    # Show some examples
    print("\nFirst 10 Word-NER Examples:")
    for word, ner_tag in ner_stats['word_ner_pairs'][:10]:
        desc = NER_DESCRIPTIONS.get(ner_tag, "Unknown")
        print(f"{word}: {ner_tag} ({desc})")