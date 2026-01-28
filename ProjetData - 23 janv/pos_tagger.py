import nltk
from collections import Counter
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def get_word_pos_pairs(sentences):
    """Get (word, POS) pairs for all sentences"""
    word_pos_pairs = []
    for sentence in sentences:
        tokens = nltk.word_tokenize(sentence)
        tagged = nltk.pos_tag(tokens)
        word_pos_pairs.extend(tagged)
    return word_pos_pairs

def get_pos_tags_only(sentences):
    """Get only POS tags for all sentences"""
    word_pos_pairs = get_word_pos_pairs(sentences)
    return [pos for _, pos in word_pos_pairs]

def get_pos_frequencies(sentences):
    """Get POS tag frequencies"""
    pos_tags = get_pos_tags_only(sentences)
    return Counter(pos_tags)

def get_pos_bigram_frequencies(sentences):
    """Get POS bigram frequencies"""
    pos_tags = get_pos_tags_only(sentences)
    bigrams = list(nltk.bigrams(pos_tags))
    return Counter(bigrams)

def load_sentences(path):
    """Load sentences from file"""
    with open(path, encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

if __name__ == "__main__":
    sentences = load_sentences('data/sentences.txt')
    
    # Get POS statistics
    pos_freq = get_pos_frequencies(sentences)
    pos_bigram_freq = get_pos_bigram_frequencies(sentences)
    
    print("Top 10 POS tags:", pos_freq.most_common(10))
    print("Top 10 POS bigrams:", pos_bigram_freq.most_common(10))