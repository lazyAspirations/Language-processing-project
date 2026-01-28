# analyze_sentences.py

import re
import os
from topic_model import TopicModeler

def load_sentences(file_path):
    """
    Load sentences from file
    
    Args:
        file_path (str): Path to the text file
        
    Returns:
        list: List of sentences
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Split by sentence boundaries (period, exclamation, question mark)
    sentences = re.split(r'[.!?]+', content)
    
    # Clean up sentences (remove whitespace, filter out empty/very short sentences)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    
    return sentences

def display_document_topics(texts, topic_model):
    """
    Display topic distribution for sample documents
    
    Args:
        texts (list): List of text documents
        topic_model (TopicModeler): Fitted topic model
    """
    doc_topics = topic_model.get_document_topics(texts)
    
    print("=" * 60)
    print("SAMPLE DOCUMENT TOPIC DISTRIBUTIONS:")
    print("=" * 60)
    
    # Show first 3 documents as examples
    for i in range(min(3, len(texts))):
        print(f"\nDocument {i+1} (first 100 chars):")
        print(f"\"{texts[i][:100]}...\"")
        print("Topic probabilities:")
        
        for topic_idx, prob in enumerate(doc_topics[i]):
            print(f"  Topic {topic_idx}: {prob:.2%}")
        
        # Get dominant topic
        dominant_topic = doc_topics[i].argmax()
        print(f"  → Dominant: Topic {dominant_topic} ({doc_topics[i][dominant_topic]:.2%})")
    
    print("\n" + "=" * 60)

def main():
    # Vérifier si le fichier existe
    file_path = 'sentences_tm.txt'
    
    if not os.path.exists(file_path):
        # Essayer dans le dossier data
        file_path = os.path.join('data', 'sentences_tm.txt')
        
    if not os.path.exists(file_path):
        print(f"ERREUR : Fichier 'sentences_tm.txt' introuvable !")
        print("Veuillez d'abord exécuter preprocessing.py pour générer ce fichier.")
        return
    
    # Load sentences from file
    print(f"Chargement des phrases depuis '{file_path}'...")
    sentences = load_sentences(file_path)
    print(f"✓ {len(sentences)} phrases chargées")
    
    if len(sentences) < 10:
        print("AVERTISSEMENT : Trop peu de phrases pour un bon topic modeling!")
        print("Essayez d'extraire plus d'abstracts dans preprocessing.py")
    
    # Create and fit topic model
    print("\nEntraînement du modèle LDA avec 3 topics...")
    topic_model = TopicModeler(n_topics=3, random_state=42)
    topic_model.fit_lda(sentences)
    
    # Display topics with top words
    print("\n" + "=" * 60)
    print("TOPICS AVEC MOTS CLÉS:")
    print("=" * 60)
    topic_model.display_topics(n_words=20)
    
    # Get dominant topics
    print("=" * 60)
    print("TOPICS LES PLUS DOMINANTS DANS TOUS LES DOCUMENTS:")
    print("=" * 60)
    dominant_topics = topic_model.get_dominant_topics(sentences, top_n=3)
    
    for rank, (topic_idx, prob) in enumerate(dominant_topics, 1):
        print(f"\n#{rank} Topic le plus cité: Topic {topic_idx}")
        print(f"  Probabilité moyenne sur tous les documents: {prob:.2%}")
        
        # Show top words for this topic
        topics_words = topic_model.get_top_words(n_words=10)
        print(f"  Top mots: {' | '.join(topics_words[topic_idx][:10])}")
    
    # Show sample document distributions
    display_document_topics(sentences, topic_model)
    
    # Additional analysis: Show document counts per dominant topic
    doc_topics = topic_model.get_document_topics(sentences)
    dominant_per_doc = doc_topics.argmax(axis=1)
    
    print("\nCOMPTAGE DE DOCUMENTS PAR TOPIC DOMINANT:")
    for topic_idx in range(3):
        count = sum(dominant_per_doc == topic_idx)
        percentage = count / len(sentences) * 100
        print(f"  Topic {topic_idx}: {count} documents ({percentage:.1f}%)")
    
    # Show examples for each topic
    print("\n" + "=" * 60)
    print("EXEMPLES POUR CHAQUE TOPIC:")
    print("=" * 60)
    
    topics_words = topic_model.get_top_words(n_words=5)
    for topic_idx in range(3):
        print(f"\nTopic {topic_idx} (mots: {', '.join(topics_words[topic_idx][:5])}):")
        
        # Trouver 2 documents dominés par ce topic
        topic_docs = []
        for i in range(len(sentences)):
            if dominant_per_doc[i] == topic_idx and len(topic_docs) < 2:
                topic_docs.append(i)
        
        for doc_idx in topic_docs:
            preview = sentences[doc_idx][:80] + "..." if len(sentences[doc_idx]) > 80 else sentences[doc_idx]
            topic_probs = doc_topics[doc_idx]
            print(f"  • Doc {doc_idx}: {topic_probs[topic_idx]:.1%} | \"{preview}\"")

if __name__ == "__main__":
    main()