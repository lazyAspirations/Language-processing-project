from collections import Counter
import nltk
nltk.download('punkt')

def tokenize_sentences(sentences):
    """Tokenize all sentences into words"""
    tokens = []
    for sent in sentences:
        tokens.extend(nltk.word_tokenize(sent.lower()))
    return tokens

def compute_unigram_frequencies(tokens):
    """Compute unigram frequencies"""
    return Counter(tokens)

def compute_unigram_probabilities(tokens):
    """Compute unigram probabilities"""
    total = len(tokens)
    freqs = Counter(tokens)
    probs = {word: freq / total for word, freq in freqs.items()}
    return freqs, probs

def compute_bigram_frequencies(tokens):
    """Compute bigram frequencies"""
    bigrams = list(nltk.bigrams(tokens))
    return Counter(bigrams)

def compute_bigram_probabilities(tokens):
    """Compute bigram probabilities"""
    bigrams = list(nltk.bigrams(tokens))
    total = len(bigrams)
    freqs = Counter(bigrams)
    probs = {bg: freq / total for bg, freq in freqs.items()}
    return freqs, probs

def compute_trigram_frequencies(tokens):
    """Compute trigram frequencies"""
    trigrams = list(nltk.trigrams(tokens))
    return Counter(trigrams)

def compute_trigram_probabilities(tokens):
    """Compute trigram probabilities"""
    trigrams = list(nltk.trigrams(tokens))
    total = len(trigrams)
    freqs = Counter(trigrams)
    probs = {tg: freq / total for tg, freq in freqs.items()}
    return freqs, probs

def get_all_words_with_pos(sentences):
    """Get all words with their POS tags"""
    all_words_with_pos = []
    for sentence in sentences:
        tokens = nltk.word_tokenize(sentence)
        tagged = nltk.pos_tag(tokens)
        all_words_with_pos.extend(tagged)
    return all_words_with_pos

def get_pos_statistics(sentences):
    """Get comprehensive POS statistics"""
    all_words_with_pos = get_all_words_with_pos(sentences)
    
    # Extract just the POS tags
    pos_tags = [pos for _, pos in all_words_with_pos]
    
    # Unigram POS frequencies
    pos_unigram_freq = Counter(pos_tags)
    
    # Bigram POS frequencies
    pos_bigrams = list(nltk.bigrams(pos_tags))
    pos_bigram_freq = Counter(pos_bigrams)
    
    return {
        'all_words_with_pos': all_words_with_pos,
        'pos_tags': pos_tags,
        'pos_unigram_freq': pos_unigram_freq,
        'pos_bigram_freq': pos_bigram_freq
    }

def load_sentences(path):
    """Load sentences from file"""
    with open(path, encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

if __name__ == "__main__":
    sentences = load_sentences('data/sentences.txt')
    tokens = tokenize_sentences(sentences)
    
    # Test all functions
    uni_freqs, uni_probs = compute_unigram_probabilities(tokens)
    bi_freqs, bi_probs = compute_bigram_probabilities(tokens)
    tri_freqs, tri_probs = compute_trigram_probabilities(tokens)
    
    print("Top 10 unigrams:", uni_freqs.most_common(10))
    print("Top 10 bigrams:", bi_freqs.most_common(10))
    print("Top 10 trigrams:", tri_freqs.most_common(10))
    
    # Test POS statistics
    pos_stats = get_pos_statistics(sentences)
    print("\nTop 10 POS tags:", pos_stats['pos_unigram_freq'].most_common(10))
    print("Top 10 POS bigrams:", pos_stats['pos_bigram_freq'].most_common(10))