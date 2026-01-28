# streamlit_app.py
import streamlit as st
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from preprocessing import preprocess_pubmed
from morphosyntax import enhanced_parse_with_visualization, POS_DESCRIPTIONS
from language_model import (
    load_sentences, tokenize_sentences, 
    compute_unigram_frequencies, compute_bigram_frequencies,
    compute_unigram_probabilities, compute_bigram_probabilities,
    get_pos_statistics
)
from pos_tagger import get_word_pos_pairs, get_pos_frequencies, get_pos_bigram_frequencies
from ner import get_word_ner_pairs, get_ner_frequencies, get_ner_bigram_frequencies, NER_DESCRIPTIONS, get_ner_statistics
from topic_model import TopicModeler, get_document_topic_distribution, get_topic_summary_statistics, lda_topics
from stats import compute_stats

DATA_DIR = 'data'
XML_PATH = os.path.join(DATA_DIR, 'pubmed.xml')
SENT_PATH = os.path.join(DATA_DIR, 'sentences.txt')

st.set_page_config(page_title="NLU PubMed App", layout="wide")
st.title("Natural Language Understanding Application")

# --- Helper functions for text analysis ---
def analyze_pos_for_text(text):
    """Analyze POS for a single text"""
    sentences_list = [text] if text.strip() else []
    if not sentences_list:
        return None
    
    # Get word-POS pairs
    word_pos_pairs = get_word_pos_pairs(sentences_list)
    
    # Create DataFrame
    word_pos_data = []
    for word, pos in word_pos_pairs:
        description = POS_DESCRIPTIONS.get(pos, "Unknown")
        word_pos_data.append({
            'Word': word,
            'POS Tag': pos,
            'POS Description': description
        })
    
    word_pos_df = pd.DataFrame(word_pos_data)
    return word_pos_df

def analyze_ner_for_text(text):
    """Analyze NER for a single text"""
    sentences_list = [text] if text.strip() else []
    if not sentences_list:
        return None
    
    # Get comprehensive NER statistics
    ner_stats = get_ner_statistics(sentences_list)
    word_ner_pairs = ner_stats['word_ner_pairs']
    
    # Create DataFrame for word-level NER
    word_ner_data = []
    for word, ner_tag in word_ner_pairs:
        description = NER_DESCRIPTIONS.get(ner_tag, "Unknown")
        word_ner_data.append({
            'Word': word,
            'NER Tag': ner_tag,
            'NER Description': description
        })
    
    word_ner_df = pd.DataFrame(word_ner_data)
    return word_ner_df

def analyze_topics_for_text(text, n_topics=3, n_words=10):
    """Analyze topics for a single text"""
    sentences_list = [text] if text.strip() else []
    if not sentences_list:
        return None, None
    
    try:
        # Create and fit topic model
        modeler = TopicModeler(n_topics=n_topics)
        
        # Update vectorizer configuration for small text
        modeler.vectorizer = CountVectorizer(
            ngram_range=(1, 2),
            stop_words='english',
            token_pattern=r'\b[a-z][a-z]+\b',
            lowercase=True,
            max_features=500
        )
        
        # Fit the model
        X = modeler.vectorizer.fit_transform(sentences_list)
        modeler.feature_names = modeler.vectorizer.get_feature_names_out()
        
        modeler.lda_model = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            learning_method='online'
        )
        modeler.lda_model.fit(X)
        
        # Get topic statistics
        topic_summary = get_topic_summary_statistics(modeler, n_words=n_words)
        
        # Create a nice table for topics
        topic_data = []
        for topic_idx, stats in topic_summary.items():
            topic_data.append({
                'Topic ID': topic_idx + 1,
                'Inferred Category': stats['inferred_name'],
                'Top Words': ', '.join(stats['top_words'][:n_words]),
                'Key Terms': ', '.join(stats['top_5_words'])
            })
        
        topics_df = pd.DataFrame(topic_data)
        
        # Get document-topic distribution
        doc_topics = modeler.get_document_topics(sentences_list)
        
        # Create document-topic assignment
        doc_topic_data = []
        for i, probs in enumerate(doc_topics):
            dominant_topic = np.argmax(probs)
            doc_topic_data.append({
                'Document': f"Text {i+1}",
                'Dominant Topic': dominant_topic + 1,
                'Dominant Topic Probability': f"{probs[dominant_topic]:.2%}",
                **{f'Topic {j+1}': f"{prob:.2%}" for j, prob in enumerate(probs)}
            })
        
        doc_topic_df = pd.DataFrame(doc_topic_data)
        
        return topics_df, doc_topic_df
        
    except Exception as e:
        st.warning(f"Could not perform topic modeling on this text: {str(e)}")
        return None, None

def compute_possible_generations(text):
    """Compute number of possible generations for text"""
    sentences_list = [text] if text.strip() else []
    if not sentences_list:
        return 0
    
    tokens = tokenize_sentences(sentences_list)
    return len(tokens)

# --- 1. PREPROCESSING SECTION ---
st.header("1. Preprocessing")

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
    st.warning(f"Created {DATA_DIR} directory. Please upload your pubmed.xml file.")

if os.path.exists(XML_PATH):
    st.success(f"Found pubmed.xml file in {DATA_DIR}/ directory")
    
    if os.path.exists(SENT_PATH):
        st.info("Sentences file already exists. You can reprocess if needed.")
        if st.button("Reprocess PubMed XML"):
            with st.spinner("Processing PubMed XML file..."):
                try:
                    preprocess_pubmed(XML_PATH, os.path.join(DATA_DIR, 'sentences'), min_abstracts=10)
                    st.success("Preprocessing completed successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error during preprocessing: {e}")
    else:
        if st.button("Process PubMed XML"):
            with st.spinner("Processing PubMed XML file..."):
                try:
                    preprocess_pubmed(XML_PATH, os.path.join(DATA_DIR, 'sentences'), min_abstracts=10)
                    st.success("Preprocessing completed successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error during preprocessing: {e}")
else:
    st.warning(f"Please upload a pubmed.xml file to the {DATA_DIR}/ directory")
    st.write(f"Expected file location: {XML_PATH}")

# --- Main Analysis (if sentences.txt exists) ---
if os.path.exists(SENT_PATH):
    # Load data
    sentences = load_sentences(SENT_PATH)
    
    # --- 2. PREPROCESSING RESULT ---
    st.header("2. Preprocessing Result")
    
    # Show file preview
    with st.expander("Show sentences.txt (one sentence per line)"):
        with open(SENT_PATH, encoding="utf-8") as f:
            st.code(f.read(), language="text")
    
    # Also show the topic modeling version if it exists
    tm_path = os.path.join(DATA_DIR, 'sentences_tm.txt')
    if os.path.exists(tm_path):
        with st.expander("Show sentences_tm.txt (preprocessed for topic modeling)"):
            with open(tm_path, encoding="utf-8") as f:
                st.code(f.read(), language="text")
    
    # --- 3. WORD N-GRAM ANALYSIS ---
    st.header("3. Word N-Gram Analysis")
    
    # Tokenize sentences
    tokens = tokenize_sentences(sentences)
    
    # 3.1 Unigram Frequencies
    st.subheader("3.1 Unigram Frequencies")
    unigram_freq = compute_unigram_frequencies(tokens)
    unigram_freq_df = pd.DataFrame(
        unigram_freq.most_common(), 
        columns=['Unigram', 'Frequency']
    )
    
    st.write(f"**Total unique unigrams:** {len(unigram_freq)}")
    st.write(f"**Most frequent unigram:** '{unigram_freq_df.iloc[0]['Unigram']}' ({unigram_freq_df.iloc[0]['Frequency']} times)")
    
    with st.expander("Show All Unigram Frequencies"):
        st.dataframe(unigram_freq_df, use_container_width=True)
    
    # 3.2 Unigram Probabilities
    st.subheader("3.2 Unigram Probabilities")
    unigram_freq_counter, unigram_probs = compute_unigram_probabilities(tokens)
    unigram_probs_df = pd.DataFrame(
        [(word, prob) for word, prob in unigram_probs.items()], 
        columns=['Unigram', 'Probability']
    ).sort_values('Probability', ascending=False)
    
    with st.expander("Show All Unigram Probabilities"):
        st.dataframe(unigram_probs_df, use_container_width=True)
    
    # 3.3 Bigram Frequencies
    st.subheader("3.3 Bigram Frequencies")
    bigram_freq = compute_bigram_frequencies(tokens)
    bigram_freq_df = pd.DataFrame(
        [(' '.join(bigram), freq) for bigram, freq in bigram_freq.most_common()], 
        columns=['Bigram', 'Frequency']
    )
    
    st.write(f"**Total unique bigrams:** {len(bigram_freq)}")
    if len(bigram_freq_df) > 0:
        st.write(f"**Most frequent bigram:** '{bigram_freq_df.iloc[0]['Bigram']}' ({bigram_freq_df.iloc[0]['Frequency']} times)")
    
    with st.expander("Show All Bigram Frequencies"):
        st.dataframe(bigram_freq_df, use_container_width=True)
    
    # 3.4 Bigram Probabilities
    st.subheader("3.4 Bigram Probabilities")
    bigram_freq_counter, bigram_probs = compute_bigram_probabilities(tokens)
    bigram_probs_df = pd.DataFrame(
        [(' '.join(bigram), prob) for bigram, prob in bigram_probs.items()], 
        columns=['Bigram', 'Probability']
    ).sort_values('Probability', ascending=False)
    
    with st.expander("Show All Bigram Probabilities"):
        st.dataframe(bigram_probs_df, use_container_width=True)
    
    # --- 4. DATASET SUMMARY ---
    st.header("4. Dataset Summary")
    
    stats = compute_stats(sentences)
    
    summary_table = pd.DataFrame({
        "Statistic": [
            "Number of sentences",
            "Number of words",
            "Most frequent word",
            "Most frequent word count",
            "Number of possible generations",
            "Total unique unigrams",
            "Total unique bigrams"
        ],
        "Value": [
            str(stats['num_sentences']),
            str(stats['num_words']),
            str(unigram_freq_df.iloc[0]['Unigram']) if len(unigram_freq_df) > 0 else "N/A",
            str(unigram_freq_df.iloc[0]['Frequency']) if len(unigram_freq_df) > 0 else "N/A",
            str(len(tokens)),
            str(len(unigram_freq)),
            str(len(bigram_freq))
        ]
    })
    
    st.table(summary_table)
    
    # Show top 10 bigrams in a nice table
    st.subheader("Top 10 Bigrams")
    if len(bigram_freq_df) > 0:
        top_10_bigrams = bigram_freq_df.head(10).copy()
        top_10_bigrams.index = range(1, len(top_10_bigrams) + 1)
        st.table(top_10_bigrams)
    
    # --- 5. WORD-LEVEL POS ANALYSIS ---
    st.header("5. Word-Level POS Analysis")
    
    # Get word-POS pairs
    word_pos_pairs = get_word_pos_pairs(sentences)
    
    # Create DataFrame
    word_pos_data = []
    for word, pos in word_pos_pairs:
        description = POS_DESCRIPTIONS.get(pos, "Unknown")
        word_pos_data.append({
            'Word': word,
            'POS Tag': pos,
            'POS Description': description
        })
    
    word_pos_df = pd.DataFrame(word_pos_data)
    
    st.write(f"**Total words analyzed:** {len(word_pos_df)}")
    
    # Show examples
    st.subheader("Examples (Word: POS Tag - Description)")
    examples = []
    for i, (word, pos) in enumerate(word_pos_pairs[:20]):
        description = POS_DESCRIPTIONS.get(pos, "Unknown")
        examples.append(f"{word}: {pos} - {description}")
    
    col1, col2 = st.columns(2)
    with col1:
        for i in range(0, min(10, len(examples))):
            st.write(f"• {examples[i]}")
    with col2:
        for i in range(10, min(20, len(examples))):
            st.write(f"• {examples[i]}")
    
    with st.expander("Show Complete Word-POS Table"):
        st.dataframe(word_pos_df, use_container_width=True)
    
    # --- 6. POS UNIGRAM FREQUENCIES ---
    st.header("6. POS Unigram Frequencies")
    
    pos_freq = get_pos_frequencies(sentences)
    pos_freq_data = []
    for pos_tag, freq in pos_freq.most_common():
        description = POS_DESCRIPTIONS.get(pos_tag, "Unknown")
        percentage = (freq / sum(pos_freq.values())) * 100
        pos_freq_data.append({
            'POS Tag': pos_tag,
            'Description': description,
            'Frequency': freq,
            'Percentage': f"{percentage:.2f}%"
        })
    
    pos_freq_df = pd.DataFrame(pos_freq_data)
    
    st.write(f"**Total unique POS tags:** {len(pos_freq)}")
    st.write(f"**Most frequent POS:** '{pos_freq_df.iloc[0]['POS Tag']}' ({pos_freq_df.iloc[0]['Description']}) - {pos_freq_df.iloc[0]['Frequency']} times")
    
    st.dataframe(pos_freq_df, use_container_width=True)
    
    # --- 7. POS BIGRAM FREQUENCIES ---
    st.header("7. POS Bigram Frequencies")
    
    pos_bigram_freq = get_pos_bigram_frequencies(sentences)
    pos_bigram_data = []
    for (pos1, pos2), freq in pos_bigram_freq.most_common():
        desc1 = POS_DESCRIPTIONS.get(pos1, "Unknown")
        desc2 = POS_DESCRIPTIONS.get(pos2, "Unknown")
        pos_bigram_data.append({
            'POS Bigram': f"{pos1} → {pos2}",
            'Description': f"{desc1} → {desc2}",
            'Frequency': freq
        })
    
    pos_bigram_df = pd.DataFrame(pos_bigram_data)
    
    if len(pos_bigram_df) > 0:
        st.write(f"**Total unique POS bigrams:** {len(pos_bigram_freq)}")
        st.write(f"**Most frequent POS bigram:** '{pos_bigram_df.iloc[0]['POS Bigram']}' - {pos_bigram_df.iloc[0]['Frequency']} times")
        
        st.dataframe(pos_bigram_df, use_container_width=True)
    else:
        st.write("No POS bigrams found (insufficient data)")
    
    # --- 8. NER ANALYSIS ---
    st.header("8. NER Analysis")
    
    # Get comprehensive NER statistics using the ner.py module
    ner_stats = get_ner_statistics(sentences)
    
    # 8.1 Word-Level NER Tagging
    st.subheader("8.1 Word-Level NER Tagging")
    
    word_ner_pairs = ner_stats['word_ner_pairs']
    
    # Create DataFrame for word-level NER
    word_ner_data = []
    for word, ner_tag in word_ner_pairs[:100]:  # Show first 100
        description = NER_DESCRIPTIONS.get(ner_tag, "Unknown")
        word_ner_data.append({
            'Word': word,
            'NER Tag': ner_tag,
            'NER Description': description
        })
    
    word_ner_df = pd.DataFrame(word_ner_data)
    
    st.write(f"**Total words analyzed for NER:** {len(word_ner_pairs)}")
    
    # Show examples
    st.write("**Examples (Word: NER Tag - Description):**")
    examples = []
    for i, (word, ner_tag) in enumerate(word_ner_pairs[:20]):
        description = NER_DESCRIPTIONS.get(ner_tag, "Unknown")
        examples.append(f"{word}: {ner_tag} - {description}")
    
    col1, col2 = st.columns(2)
    with col1:
        for i in range(0, min(10, len(examples))):
            st.write(f"• {examples[i]}")
    with col2:
        for i in range(10, min(20, len(examples))):
            st.write(f"• {examples[i]}")
    
    with st.expander("Show Word-NER Table (first 100)"):
        st.dataframe(word_ner_df, use_container_width=True)
    
    # 8.2 NER Unigram Frequencies with Descriptions
    st.subheader("8.2 NER Tag Frequencies")
    
    ner_freq = ner_stats['ner_frequencies']
    
    # Create DataFrame with descriptions
    ner_freq_data = []
    for ner_tag, freq in ner_freq.most_common():
        description = NER_DESCRIPTIONS.get(ner_tag, "Unknown")
        percentage = (freq / ner_stats['total_ner_tags']) * 100
        ner_freq_data.append({
            'NER Tag': ner_tag,
            'Description': description,
            'Frequency': freq,
            'Percentage': f"{percentage:.2f}%"
        })
    
    ner_freq_df = pd.DataFrame(ner_freq_data)
    
    st.write(f"**Total unique NER tags:** {ner_stats['unique_ner_tags']}")
    st.write(f"**Most frequent NER tag:** '{ner_freq_df.iloc[0]['NER Tag']}' ({ner_freq_df.iloc[0]['Description']}) - {ner_freq_df.iloc[0]['Frequency']} times")
    st.write(f"**Total tokens analyzed:** {ner_stats['total_ner_tags']}")
    
    st.dataframe(ner_freq_df, use_container_width=True)
    
    # 8.3 NER Bigram Frequencies
    st.subheader("8.3 NER Bigram Frequencies")
    
    ner_bigram_freq = ner_stats['ner_bigram_frequencies']
    
    if ner_bigram_freq:
        ner_bigram_data = []
        for (ner1, ner2), freq in ner_bigram_freq.most_common():
            desc1 = NER_DESCRIPTIONS.get(ner1, "Unknown")
            desc2 = NER_DESCRIPTIONS.get(ner2, "Unknown")
            ner_bigram_data.append({
                'NER Bigram': f"{ner1} → {ner2}",
                'Description': f"{desc1} → {desc2}",
                'Frequency': freq
            })
        
        ner_bigram_df = pd.DataFrame(ner_bigram_data)
        
        st.write(f"**Total unique NER bigrams:** {len(ner_bigram_freq)}")
        
        if len(ner_bigram_df) > 0:
            st.write(f"**Most frequent NER bigram:** '{ner_bigram_df.iloc[0]['NER Bigram']}' - {ner_bigram_df.iloc[0]['Frequency']} times")
        
        st.dataframe(ner_bigram_df, use_container_width=True)
    else:
        st.write("No NER bigrams found (insufficient data)")
    
    # --- 9. TOPIC MODELING (ENHANCED) ---
    st.header("9. Topic Modeling (LDA)")
    
    # Add configuration options
    col1, col2, col3 = st.columns(3)
    with col1:
        n_topics = st.slider("Number of Topics", min_value=2, max_value=8, value=3, key="pubmed_topics")
    with col2:
        n_words = st.slider("Words per Topic", min_value=5, max_value=20, value=10, key="pubmed_words")
    with col3:
        max_features = st.slider("Max Features", min_value=500, max_value=2000, value=1000, step=100, key="pubmed_features")

    if st.button("Run Topic Modeling", type="primary", key="pubmed_button"):
        with st.spinner("Running LDA Topic Modeling..."):
            try:
                # Create and fit topic model
                modeler = TopicModeler(n_topics=n_topics)
                
                # Update vectorizer configuration
                modeler.vectorizer = CountVectorizer(
                    ngram_range=(1, 2),
                    stop_words='english',
                    token_pattern=r'\b[a-z][a-z]+\b',
                    lowercase=True,
                    max_features=max_features
                )
                
                # Fit the model
                X = modeler.vectorizer.fit_transform(sentences)
                modeler.feature_names = modeler.vectorizer.get_feature_names_out()
                
                modeler.lda_model = LatentDirichletAllocation(
                    n_components=n_topics,
                    random_state=42,
                    learning_method='online'
                )
                modeler.lda_model.fit(X)
                
                # Get topic statistics
                topic_summary = get_topic_summary_statistics(modeler, n_words=n_words)
                
                # 9.1 Display Topics with Inferred Names
                st.subheader("9.1 Discovered Topics")
                
                # Create a nice table for topics
                topic_data = []
                for topic_idx, stats in topic_summary.items():
                    topic_data.append({
                        'Topic ID': topic_idx + 1,
                        'Inferred Category': stats['inferred_name'],
                        'Top Words': ', '.join(stats['top_words']),
                        'Key Terms': ', '.join(stats['top_5_words'])
                    })
                
                topics_df = pd.DataFrame(topic_data)
                st.dataframe(topics_df, width='stretch')
                
                # 9.2 Topic Distribution Pie Chart
                st.subheader("9.2 Topic Distribution Across Documents")
                
                # Get document-topic distribution
                doc_topics = modeler.get_document_topics(sentences)
                
                # Calculate average topic proportions
                avg_topic_probs = np.mean(doc_topics, axis=0) * 100
                
                # Create pie chart
                fig, ax = plt.subplots(figsize=(8, 6))
                colors = plt.cm.Set3(np.linspace(0, 1, n_topics))
                
                # Prepare labels with percentages
                labels = []
                for i in range(n_topics):
                    inferred_name = topic_summary[i]['inferred_name']
                    percentage = avg_topic_probs[i]
                    labels.append(f"Topic {i+1}: {inferred_name}\n({percentage:.1f}%)")
                
                wedges, texts, autotexts = ax.pie(
                    avg_topic_probs,
                    labels=labels,
                    colors=colors,
                    autopct='%1.1f%%',
                    startangle=90,
                    pctdistance=0.85
                )
                
                # Improve text appearance
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                
                ax.axis('equal')
                plt.title('Topic Distribution Across All Documents', fontsize=14, fontweight='bold')
                
                st.pyplot(fig)
                
                # 9.3 Document-Topic Assignment Table
                st.subheader("9.3 Sample Document-Topic Assignments")
                
                doc_topic_df = get_document_topic_distribution(sentences, modeler)
                
                # Show first 10 documents
                st.write(f"**First {min(10, len(doc_topic_df))} documents with topic assignments:**")
                
                display_cols = ['Document_Preview', 'Dominant_Topic', 'Dominant_Topic_Proba'] + \
                              [f'Topic_{i}' for i in range(n_topics)]
                
                display_df = doc_topic_df[display_cols].head(10).copy()
                display_df['Dominant_Topic'] = display_df['Dominant_Topic'] + 1
                display_df['Dominant_Topic_Proba'] = display_df['Dominant_Topic_Proba'].apply(lambda x: f"{x:.2%}")
                
                # Format topic probabilities
                for i in range(n_topics):
                    display_df[f'Topic_{i}'] = display_df[f'Topic_{i}'].apply(lambda x: f"{x:.2%}")
                
                st.dataframe(display_df, width='stretch')
                
            except Exception as e:
                st.error(f"Error in topic modeling: {str(e)}")
                st.write("Trying with default settings...")
                
                # Fallback to simple display
                try:
                    topics = lda_topics(sentences, n_topics=n_topics, n_words=n_words)
                    for i, topic_words in enumerate(topics):
                        st.write(f"**Topic {i+1}:** {', '.join(topic_words)}")
                except:
                    st.write("Could not perform topic modeling. Please check your data.")
    
    # --- 10. USER TEXT ANALYSIS ---
    st.header("10. Analyze Your Text")
    
    input_type = st.radio("Choose input type:", ("Sentence", "Paragraph", "Upload File"), horizontal=True, key="input_type")
    user_text = ""
    
    if input_type == "Sentence":
        user_text = st.text_input("Enter a sentence:", key="sentence_input")
        if user_text:
            st.subheader("Analysis Results for Your Sentence")
            
            # 1. Enhanced Parsing Analysis (existing)
            enhanced_parse_with_visualization(user_text)
            
            # 2. Word-Level POS Analysis
            st.subheader("Word-Level POS Analysis")
            pos_df = analyze_pos_for_text(user_text)
            if pos_df is not None and not pos_df.empty:
                st.write(f"**Total words analyzed:** {len(pos_df)}")
                st.dataframe(pos_df, use_container_width=True)
                
                # Show POS summary
                st.write("**POS Summary:**")
                pos_counts = pos_df['POS Tag'].value_counts()
                for pos, count in pos_counts.items():
                    desc = POS_DESCRIPTIONS.get(pos, "Unknown")
                    percentage = (count / len(pos_df)) * 100
                    st.write(f"• {pos} ({desc}): {count} words ({percentage:.1f}%)")
            
            # 3. Word-Level NER Tagging
            st.subheader("Word-Level NER Tagging")
            ner_df = analyze_ner_for_text(user_text)
            if ner_df is not None and not ner_df.empty:
                st.write(f"**Total words analyzed:** {len(ner_df)}")
                st.dataframe(ner_df, use_container_width=True)
                
                # Show NER summary
                st.write("**NER Summary:**")
                ner_counts = ner_df['NER Tag'].value_counts()
                for ner_tag, count in ner_counts.items():
                    if ner_tag != 'O':  # Skip 'O' (outside of named entity)
                        desc = NER_DESCRIPTIONS.get(ner_tag, "Unknown")
                        percentage = (count / len(ner_df)) * 100
                        st.write(f"• {ner_tag} ({desc}): {count} entities ({percentage:.1f}%)")
            
            # 4. Topic Modeling (3 main topics)
            st.subheader("Topic Analysis (3 Main Topics)")
            topics_df, doc_topic_df = analyze_topics_for_text(user_text, n_topics=3, n_words=10)
            if topics_df is not None:
                st.dataframe(topics_df, use_container_width=True)
                
                # Show document-topic assignment
                if doc_topic_df is not None:
                    st.write("**Document-Topic Assignment:**")
                    st.dataframe(doc_topic_df, use_container_width=True)
            else:
                st.warning("Text is too short for topic modeling. Please enter a longer text.")
            
    elif input_type == "Paragraph":
        user_text = st.text_area("Enter a paragraph (max 500 words):", height=150, key="paragraph_input")
        if user_text:
            st.subheader("Analysis Results for Your Paragraph")
            
            # 1. Number of Possible Generations
            possible_generations = compute_possible_generations(user_text)
            st.write(f"**Number of possible generations:** {possible_generations}")
            
            # Tokenize and get basic stats
            user_tokens = tokenize_sentences([user_text])
            user_unigram_freq = compute_unigram_frequencies(user_tokens)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Word Count", len(user_tokens))
                
            with col2:
                st.metric("Unique Words", len(user_unigram_freq))
                
            with col3:
                if user_unigram_freq:
                    most_common = max(user_unigram_freq.items(), key=lambda x: x[1])
                    st.metric("Most Frequent Word", most_common[0])
            
            # 2. Word-Level POS Analysis
            st.subheader("Word-Level POS Analysis")
            pos_df = analyze_pos_for_text(user_text)
            if pos_df is not None and not pos_df.empty:
                st.write(f"**Total words analyzed:** {len(pos_df)}")
                
                # Show top 20 entries
                st.dataframe(pos_df.head(20), use_container_width=True)
                
                # Show POS summary
                st.write("**POS Summary:**")
                pos_counts = pos_df['POS Tag'].value_counts()
                pos_summary_data = []
                for pos, count in pos_counts.items():
                    desc = POS_DESCRIPTIONS.get(pos, "Unknown")
                    percentage = (count / len(pos_df)) * 100
                    pos_summary_data.append({
                        'POS Tag': pos,
                        'Description': desc,
                        'Count': count,
                        'Percentage': f"{percentage:.1f}%"
                    })
                
                pos_summary_df = pd.DataFrame(pos_summary_data)
                st.dataframe(pos_summary_df, use_container_width=True)
            
            # 3. Word-Level NER Tagging
            st.subheader("Word-Level NER Tagging")
            ner_df = analyze_ner_for_text(user_text)
            if ner_df is not None and not ner_df.empty:
                st.write(f"**Total words analyzed:** {len(ner_df)}")
                
                # Filter out 'O' tags for better visualization
                ner_filtered_df = ner_df[ner_df['NER Tag'] != 'O']
                if not ner_filtered_df.empty:
                    st.dataframe(ner_filtered_df.head(20), use_container_width=True)
                    
                    # Show NER summary
                    st.write("**NER Summary (excluding 'O' tags):**")
                    ner_counts = ner_filtered_df['NER Tag'].value_counts()
                    ner_summary_data = []
                    for ner_tag, count in ner_counts.items():
                        desc = NER_DESCRIPTIONS.get(ner_tag, "Unknown")
                        percentage = (count / len(ner_df)) * 100
                        ner_summary_data.append({
                            'NER Tag': ner_tag,
                            'Description': desc,
                            'Count': count,
                            'Percentage': f"{percentage:.1f}%"
                        })
                    
                    ner_summary_df = pd.DataFrame(ner_summary_data)
                    st.dataframe(ner_summary_df, use_container_width=True)
                else:
                    st.info("No named entities found in the text.")
            
            # 4. Topic Modeling (3 main topics)
            st.subheader("Topic Analysis (3 Main Topics)")
            topics_df, doc_topic_df = analyze_topics_for_text(user_text, n_topics=3, n_words=10)
            if topics_df is not None:
                st.dataframe(topics_df, use_container_width=True)
                
                # Show document-topic assignment
                if doc_topic_df is not None:
                    st.write("**Document-Topic Assignment:**")
                    st.dataframe(doc_topic_df, use_container_width=True)
            else:
                st.warning("Text is too short for topic modeling. Please enter a longer text.")
    
    elif input_type == "Upload File":
        uploaded_file = st.file_uploader("Upload a text file (max 500 words)", type=["txt"], key="file_uploader")
        if uploaded_file:
            file_text = uploaded_file.read().decode("utf-8")
            
            # Limit to 500 words if needed
            words = file_text.split()
            if len(words) > 500:
                file_text = ' '.join(words[:500])
                st.warning(f"File truncated to first 500 words. Original: {len(words)} words.")
            
            st.subheader("Analysis Results for Uploaded File")
            
            # Show file content preview
            with st.expander("Show file content preview"):
                st.text(file_text[:1000] + ("..." if len(file_text) > 1000 else ""))
            
            # 1. Number of Possible Generations
            possible_generations = compute_possible_generations(file_text)
            st.write(f"**Number of possible generations:** {possible_generations}")
            
            # Basic analysis
            user_tokens = tokenize_sentences([file_text])
            user_unigram_freq = compute_unigram_frequencies(user_tokens)
            user_bigram_freq = compute_bigram_frequencies(user_tokens)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Basic Statistics:**")
                st.write(f"• Word count: {len(user_tokens)}")
                st.write(f"• Unique words: {len(user_unigram_freq)}")
                if user_unigram_freq:
                    most_common = max(user_unigram_freq.items(), key=lambda x: x[1])
                    st.write(f"• Most frequent word: '{most_common[0]}' ({most_common[1]} times)")
            
            with col2:
                # Show top 5 bigrams
                if user_bigram_freq:
                    st.write("**Top 5 Bigrams:**")
                    for bigram, freq in list(user_bigram_freq.most_common(5)):
                        st.write(f"• {' '.join(bigram)}: {freq}")
            
            # 2. Word-Level POS Analysis
            st.subheader("Word-Level POS Analysis")
            pos_df = analyze_pos_for_text(file_text)
            if pos_df is not None and not pos_df.empty:
                st.write(f"**Total words analyzed:** {len(pos_df)}")
                
                # Show POS summary
                st.write("**POS Summary (Top 10):**")
                pos_counts = pos_df['POS Tag'].value_counts().head(10)
                pos_summary_data = []
                for pos, count in pos_counts.items():
                    desc = POS_DESCRIPTIONS.get(pos, "Unknown")
                    percentage = (count / len(pos_df)) * 100
                    pos_summary_data.append({
                        'POS Tag': pos,
                        'Description': desc,
                        'Count': count,
                        'Percentage': f"{percentage:.1f}%"
                    })
                
                pos_summary_df = pd.DataFrame(pos_summary_data)
                st.dataframe(pos_summary_df, use_container_width=True)
            
            # 3. Word-Level NER Tagging
            st.subheader("Word-Level NER Tagging")
            ner_df = analyze_ner_for_text(file_text)
            if ner_df is not None and not ner_df.empty:
                st.write(f"**Total words analyzed:** {len(ner_df)}")
                
                # Filter out 'O' tags for better visualization
                ner_filtered_df = ner_df[ner_df['NER Tag'] != 'O']
                if not ner_filtered_df.empty:
                    # Show NER summary
                    st.write("**NER Summary (excluding 'O' tags):**")
                    ner_counts = ner_filtered_df['NER Tag'].value_counts()
                    ner_summary_data = []
                    for ner_tag, count in ner_counts.items():
                        desc = NER_DESCRIPTIONS.get(ner_tag, "Unknown")
                        percentage = (count / len(ner_df)) * 100
                        ner_summary_data.append({
                            'NER Tag': ner_tag,
                            'Description': desc,
                            'Count': count,
                            'Percentage': f"{percentage:.1f}%"
                        })
                    
                    ner_summary_df = pd.DataFrame(ner_summary_data)
                    st.dataframe(ner_summary_df, use_container_width=True)
                else:
                    st.info("No named entities found in the text.")
            
            # 4. Topic Modeling (3 main topics)
            st.subheader("Topic Analysis (3 Main Topics)")
            topics_df, doc_topic_df = analyze_topics_for_text(file_text, n_topics=3, n_words=10)
            if topics_df is not None:
                st.dataframe(topics_df, use_container_width=True)
                
                # Show document-topic assignment
                if doc_topic_df is not None:
                    st.write("**Document-Topic Assignment:**")
                    st.dataframe(doc_topic_df, use_container_width=True)
            else:
                st.warning("Text is too short for topic modeling. Please upload a longer text.")

else:
    st.info("Please process the PubMed XML file first using the Preprocessing section above.")