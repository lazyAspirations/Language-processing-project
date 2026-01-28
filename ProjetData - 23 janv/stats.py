import nltk
from collections import Counter
nltk.download('punkt')

def load_sentences(path):
    with open(path, encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def compute_stats(sentences):
    num_sentences = len(sentences)
    words = [word.lower() for sent in sentences for word in nltk.word_tokenize(sent)]
    num_words = len(words)
    word_freq = Counter(words)
    most_common_word, most_common_count = word_freq.most_common(1)[0]
    bigrams = list(nltk.bigrams(words))
    bigram_freq = Counter(bigrams)
    stats = {
        'num_sentences': num_sentences,
        'num_words': num_words,
        'most_common_word': most_common_word,
        'most_common_word_count': most_common_count,
        'top_10_words': word_freq.most_common(10),
        'top_10_bigrams': bigram_freq.most_common(10)
    }
    return stats

def print_stats(stats):
    print(f"Number of sentences: {stats['num_sentences']}")
    print(f"Number of words: {stats['num_words']}")
    print(f"Most frequent word: '{stats['most_common_word']}' ({stats['most_common_word_count']} times)")
    print("Top 10 words:")
    for word, count in stats['top_10_words']:
        print(f"  {word}: {count}")
    print("Top 10 bigrams:")
    for bigram, count in stats['top_10_bigrams']:
        print(f"  {bigram}: {count}")

if __name__ == "__main__":
    sentences = load_sentences('data/sentences.txt')
    stats = compute_stats(sentences)
    print_stats(stats)
