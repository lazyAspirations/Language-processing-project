# topic_model.py

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

class TopicModeler:
    def __init__(self, n_topics=3, random_state=42):
        """
        Initialize the topic modeler
        
        Args:
            n_topics (int): Number of topics to extract
            random_state (int): Random seed for reproducibility
        """
        self.n_topics = n_topics
        self.random_state = random_state
        self.vectorizer = None
        self.lda_model = None
        self.feature_names = None
        
    def preprocess_texts(self, texts):
        """
        Preprocess texts using CountVectorizer
        
        Args:
            texts (list): List of text documents
            
        Returns:
            scipy.sparse.csr_matrix: Transformed document-term matrix
        """
        # Create CountVectorizer with similar parameters to Lab 6
        self.vectorizer = CountVectorizer(
            ngram_range=(1, 2),
            stop_words='english',
            token_pattern=r'\b[a-z][a-z]+\b',
            lowercase=True,
            max_features=1000
        )
        
        X = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        return X
    
    def fit_lda(self, texts):
        """
        Fit LDA model to texts
        
        Args:
            texts (list): List of text documents
        """
        # Preprocess texts
        X = self.preprocess_texts(texts)
        
        # Create and fit LDA model
        self.lda_model = LatentDirichletAllocation(
            n_components=self.n_topics,
            random_state=self.random_state,
            learning_method='online'
        )
        
        self.lda_model.fit(X)
    
    def get_document_topics(self, texts):
        """
        Get topic distribution for each document
        
        Args:
            texts (list): List of text documents
            
        Returns:
            numpy.ndarray: Topic distribution matrix (n_documents x n_topics)
        """
        if self.vectorizer is None or self.lda_model is None:
            self.fit_lda(texts)
        
        X = self.vectorizer.transform(texts)
        return self.lda_model.transform(X)
    
    def get_top_words(self, n_words=20):
        """
        Get top words for each topic
        
        Args:
            n_words (int): Number of top words to return per topic
            
        Returns:
            dict: Dictionary mapping topic index to list of top words
        """
        if self.lda_model is None or self.feature_names is None:
            raise ValueError("Model not fitted yet. Call fit_lda() first.")
        
        topics_words = {}
        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_indices = topic.argsort()[:-n_words - 1:-1]
            top_words = [self.feature_names[i] for i in top_indices]
            topics_words[topic_idx] = top_words
        
        return topics_words
    
    def display_topics(self, n_words=20):
        """
        Display topics with their top words
        
        Args:
            n_words (int): Number of top words to display
        """
        topics_words = self.get_top_words(n_words)
        
        for topic_idx, words in topics_words.items():
            print(f"Topic {topic_idx}:")
            print("  " + " ".join(words[:10]))
            print("  " + " ".join(words[10:]))
            print()
    
    def get_dominant_topics(self, texts, top_n=3):
        """
        Get the most dominant topics across all documents
        
        Args:
            texts (list): List of text documents
            top_n (int): Number of top topics to return
            
        Returns:
            list: List of tuples (topic_index, average_probability)
        """
        doc_topics = self.get_document_topics(texts)
        
        # Calculate average probability for each topic across all documents
        avg_probabilities = np.mean(doc_topics, axis=0)
        
        # Get top N topics
        top_indices = avg_probabilities.argsort()[::-1][:top_n]
        dominant_topics = [(idx, avg_probabilities[idx]) for idx in top_indices]
        
        return dominant_topics
    
# Ajoutez cette fonction à la fin de topic_model.py (après la classe TopicModeler)

def lda_topics(sentences, n_topics=3, n_words=10):
    """
    Wrapper function for Streamlit app compatibility
    
    Args:
        sentences (list): List of text documents
        n_topics (int): Number of topics to extract
        n_words (int): Number of words per topic
    
    Returns:
        list: List of topics, each as a list of words
    """
    modeler = TopicModeler(n_topics=n_topics)
    modeler.fit_lda(sentences)
    topics_words = modeler.get_top_words(n_words=n_words)
    
    # Convert dict to list
    topics = []
    for i in range(n_topics):
        if i in topics_words:
            topics.append(topics_words[i])
        else:
            topics.append([])
    
    return topics

# Ajoutez ces fonctions à la fin de topic_model.py (après la fonction lda_topics)

def get_document_topic_distribution(texts, topic_modeler):
    """
    Get detailed document-topic distribution
    
    Args:
        texts (list): List of text documents
        topic_modeler (TopicModeler): Fitted topic modeler
    
    Returns:
        pandas.DataFrame: DataFrame with document-topic distribution
    """
    import pandas as pd
    
    doc_topics = topic_modeler.get_document_topics(texts)
    
    # Create DataFrame
    df = pd.DataFrame(doc_topics, columns=[f'Topic_{i}' for i in range(doc_topics.shape[1])])
    
    # Add document preview
    df['Document_Preview'] = [text[:100] + '...' if len(text) > 100 else text for text in texts]
    
    # Add dominant topic
    df['Dominant_Topic'] = doc_topics.argmax(axis=1)
    df['Dominant_Topic_Proba'] = doc_topics.max(axis=1)
    
    return df

def get_topic_summary_statistics(topic_modeler, n_words=10):
    """
    Get comprehensive topic summary statistics
    
    Args:
        topic_modeler (TopicModeler): Fitted topic modeler
        n_words (int): Number of top words per topic
    
    Returns:
        dict: Dictionary with topic statistics
    """
    topics_words = topic_modeler.get_top_words(n_words=n_words)
    
    summary = {}
    for topic_idx, words in topics_words.items():
        # Try to infer topic name from top words
        inferred_name = infer_topic_name(words[:5])
        
        summary[topic_idx] = {
            'inferred_name': inferred_name,
            'top_words': words,
            'top_5_words': words[:5],
            'word_count': len(words)
        }
    
    return summary

def infer_topic_name(top_words):
    """
    Infer a descriptive name for a topic based on its top words
    
    Args:
        top_words (list): List of top words for the topic
    
    Returns:
        str: Inferred topic name
    """
    # Medical/bioinformatics related keywords mapping
    medical_keywords = {
        'gene': 'Genetics',
        'protein': 'Proteins',
        'mutation': 'Mutations',
        'cancer': 'Oncology',
        'patient': 'Clinical',
        'risk': 'Risk Factors',
        'treatment': 'Therapeutics',
        'drug': 'Pharmacology',
        'cell': 'Cell Biology',
        'disease': 'Pathology',
        'clinical': 'Clinical Studies',
        'study': 'Research',
        'analysis': 'Data Analysis',
        'expression': 'Gene Expression',
        'sequence': 'Sequencing',
        'therapy': 'Therapy',
        'diagnosis': 'Diagnostics',
        'biomarker': 'Biomarkers',
        'genome': 'Genomics',
        'transcript': 'Transcriptomics'
    }
    
    # Check for medical keywords in top words
    topic_categories = []
    for word in top_words:
        for key, category in medical_keywords.items():
            if key in word.lower() and category not in topic_categories:
                topic_categories.append(category)
    
    if topic_categories:
        return " | ".join(topic_categories[:2])  # Combine up to 2 categories
    else:
        # If no medical keywords found, use generic naming
        return f"Topic: {' | '.join(top_words[:3])}"