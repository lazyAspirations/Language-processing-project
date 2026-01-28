import spacy
from spacy import displacy
import nltk
from nltk import pos_tag, word_tokenize
from nltk.tree import Tree
from nltk.chunk import RegexpParser
import matplotlib.pyplot as plt
import streamlit as st
from collections import defaultdict
import re
import io
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Streamlit

# Download required NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# POS tag mappings to human-readable names
POS_DESCRIPTIONS = {
    'CC': 'Coordinating conjunction',
    'CD': 'Cardinal number',
    'DT': 'Determiner',
    'EX': 'Existential there',
    'FW': 'Foreign word',
    'IN': 'Preposition/subordinating conjunction',
    'JJ': 'Adjective',
    'JJR': 'Adjective, comparative',
    'JJS': 'Adjective, superlative',
    'LS': 'List marker',
    'MD': 'Modal',
    'NN': 'Noun, singular or mass',
    'NNS': 'Noun, plural',
    'NNP': 'Proper noun, singular',
    'NNPS': 'Proper noun, plural',
    'PDT': 'Predeterminer',
    'POS': 'Possessive ending',
    'PRP': 'Personal pronoun',
    'PRP$': 'Possessive pronoun',
    'RB': 'Adverb',
    'RBR': 'Adverb, comparative',
    'RBS': 'Adverb, superlative',
    'RP': 'Particle',
    'SYM': 'Symbol',
    'TO': 'to',
    'UH': 'Interjection',
    'VB': 'Verb, base form',
    'VBD': 'Verb, past tense',
    'VBG': 'Verb, gerund/present participle',
    'VBN': 'Verb, past participle',
    'VBP': 'Verb, non-3rd person singular present',
    'VBZ': 'Verb, 3rd person singular present',
    'WDT': 'Wh-determiner',
    'WP': 'Wh-pronoun',
    'WP$': 'Possessive wh-pronoun',
    'WRB': 'Wh-adverb',
    ',': 'Comma',
    '.': 'Period',
    ':': 'Colon',
    ';': 'Semicolon',
    '?': 'Question mark',
    '!': 'Exclamation mark',
    '-': 'Dash',
    '(': 'Left parenthesis',
    ')': 'Right parenthesis',
    '"': 'Quotation mark',
    "'": 'Apostrophe',
    '`': 'Backtick',
    '``': 'Opening quotation mark',
    "''": 'Closing quotation mark',
    '$': 'Dollar sign',
    '#': 'Number sign'
}

def clean_sentence(sentence):
    """Clean and normalize the input sentence"""
    # Remove extra whitespace
    sentence = sentence.strip()
    
    # Remove multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)
    
    # Ensure proper spacing around punctuation (optional, can be adjusted)
    sentence = re.sub(r'\s*([.,!?;:])\s*', r' \1 ', sentence)
    
    # Remove leading/trailing punctuation
    sentence = re.sub(r'^\s*[.,!?;:]+', '', sentence)
    sentence = re.sub(r'[.,!?;:]+\s*$', '', sentence)
    
    return sentence.strip()

def nltk_parse(sentence):
    # Clean the sentence first
    cleaned_sentence = clean_sentence(sentence)
    
    # Get POS tags dynamically
    tokens = word_tokenize(cleaned_sentence)
    tagged = pos_tag(tokens)
    
    # Display POS tags with descriptions
    pos_output = " | ".join([f"{word}/{tag} ({POS_DESCRIPTIONS.get(tag, tag)})" for word, tag in tagged])
    st.write(f"**POS tagged sentence:** {pos_output}")
    
    # Create grammar dynamically based on POS tags
    grammar_rules = create_dynamic_grammar(tagged)
    
    try:
        parser = nltk.ChartParser(nltk.CFG.fromstring(grammar_rules))
        trees = list(parser.parse(tokens))
        return trees, tagged
    except Exception as e:
        st.write(f"Grammar parsing failed: {e}")
        st.write("Trying with simplified grammar...")
        return try_simplified_parsing(tagged, tokens)

def create_dynamic_grammar(tagged_tokens):
    """Create grammar rules dynamically based on POS tags in the sentence"""
    
    # Group words by their POS tags
    pos_to_words = defaultdict(set)
    for word, pos in tagged_tokens:
        pos_to_words[pos].add(word)
    
    # Start building grammar
    grammar_lines = []
    
    # S rule (sentence) - include punctuation
    grammar_lines.append("S -> NP VP | VP NP | S CONJ S | NP VP PP | VP PP | S CC S | S PUNC")
    
    # NP rules (noun phrase)
    grammar_lines.append("NP -> DT NN | DT JJ NN | DT JJ JJ NN | DT NN NN | NN | NNS | NNPS | NNP | PRP | NP PP | NP CC NP | JJ NN | CD NN")
    
    # VP rules (verb phrase)
    grammar_lines.append("VP -> VB | VBZ | VBP | VBD | VBG | VBN | VP NP | VP ADVP | VP PP | VP ADJP | MD VP | TO VP | VBZ ADJP | VBG NP")
    
    # PP rules (prepositional phrase)
    grammar_lines.append("PP -> IN NP | TO NP | IN VP")
    
    # Other phrase types
    grammar_lines.append("ADJP -> JJ | JJR | JJS | RB JJ | JJ CC JJ")
    grammar_lines.append("ADVP -> RB | RBR | RBS | WRB")
    grammar_lines.append("CONJ -> CC | IN")
    
    # Punctuation handling
    grammar_lines.append("PUNC -> '.' | ',' | '?' | '!' | ':' | ';'")
    
    # Add terminal rules for each POS tag found
    for pos_tag, words in pos_to_words.items():
        # Skip punctuation for terminal rules (handled separately)
        if pos_tag in [',', '.', ':', ';', '?', '!', '-', '(', ')', '"', "'", '`', '``', "''", '$', '#']:
            # Add punctuation to terminal rules
            quoted_words = " | ".join([f"'{w}'" for w in words])
            grammar_lines.append(f"PUNC -> {quoted_words}")
            continue
            
        # Quote each word and join with |
        quoted_words = " | ".join([f"'{w}'" for w in words])
        
        # Map NLTK tags to our grammar symbols
        if pos_tag in ['NN', 'NNS', 'NNP', 'NNPS']:
            grammar_lines.append(f"NN -> {quoted_words}")
        elif pos_tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
            # Map to appropriate verb categories
            if pos_tag == 'VB':
                grammar_lines.append(f"VB -> {quoted_words}")
            elif pos_tag == 'VBZ':
                grammar_lines.append(f"VBZ -> {quoted_words}")
            elif pos_tag == 'VBP':
                grammar_lines.append(f"VBP -> {quoted_words}")
            elif pos_tag == 'VBD':
                grammar_lines.append(f"VBD -> {quoted_words}")
            elif pos_tag == 'VBG':
                grammar_lines.append(f"VBG -> {quoted_words}")
            elif pos_tag == 'VBN':
                grammar_lines.append(f"VBN -> {quoted_words}")
        elif pos_tag == 'JJ' or pos_tag == 'JJR' or pos_tag == 'JJS':
            grammar_lines.append(f"JJ -> {quoted_words}")
        elif pos_tag == 'RB' or pos_tag == 'RBR' or pos_tag == 'RBS' or pos_tag == 'WRB':
            grammar_lines.append(f"RB -> {quoted_words}")
        elif pos_tag == 'DT':
            grammar_lines.append(f"DT -> {quoted_words}")
        elif pos_tag == 'IN':
            grammar_lines.append(f"IN -> {quoted_words}")
        elif pos_tag == 'TO':
            grammar_lines.append(f"TO -> {quoted_words}")
        elif pos_tag == 'CC':
            grammar_lines.append(f"CC -> {quoted_words}")
        elif pos_tag == 'PRP' or pos_tag == 'PRP$':
            grammar_lines.append(f"PRP -> {quoted_words}")
        elif pos_tag == 'MD':
            grammar_lines.append(f"MD -> {quoted_words}")
        elif pos_tag == 'CD':
            grammar_lines.append(f"CD -> {quoted_words}")
        elif pos_tag == 'RP':
            grammar_lines.append(f"RP -> {quoted_words}")
        elif pos_tag == 'EX':
            grammar_lines.append(f"EX -> {quoted_words}")
        elif pos_tag == 'FW':
            grammar_lines.append(f"FW -> {quoted_words}")
        elif pos_tag == 'PDT':
            grammar_lines.append(f"PDT -> {quoted_words}")
        elif pos_tag == 'POS':
            grammar_lines.append(f"POS -> {quoted_words}")
        elif pos_tag == 'SYM':
            grammar_lines.append(f"SYM -> {quoted_words}")
        elif pos_tag == 'UH':
            grammar_lines.append(f"UH -> {quoted_words}")
        elif pos_tag == 'WDT':
            grammar_lines.append(f"WDT -> {quoted_words}")
        elif pos_tag == 'WP' or pos_tag == 'WP$':
            grammar_lines.append(f"WP -> {quoted_words}")
        elif pos_tag == 'LS':
            grammar_lines.append(f"LS -> {quoted_words}")
    
    # Join all grammar rules
    grammar = "\n".join(grammar_lines)
    
    # Display the generated grammar (optional, for debugging)
    with st.expander("View Generated Grammar Rules"):
        st.code(grammar)
    
    return grammar

def try_simplified_parsing(tagged, tokens):
    """Try parsing with a more permissive grammar"""
    # Remove punctuation tokens for simplified parsing
    non_punct_tokens = []
    non_punct_tagged = []
    
    for token, tag in zip(tokens, tagged):
        word, pos = tag
        if pos not in [',', '.', '!', '?', ';', ':', '-', '(', ')', '"', "'"]:
            non_punct_tokens.append(token)
            non_punct_tagged.append(tag)
    
    if not non_punct_tokens:
        return None, tagged
    
    # Create simplified grammar
    simplified_grammar = """
    S -> NP VP | VP NP | S CC S
    NP -> DT NN | DT JJ NN | NN | NNS | NNP | PRP | NP CC NP
    VP -> VB | VBZ | VBP | VBD | VBG | VBN | VP NP | MD VP
    PP -> IN NP
    JJ -> 'JJ'
    NN -> 'NN'
    VB -> 'VB'
    DT -> 'DT'
    IN -> 'IN'
    CC -> 'CC'
    PRP -> 'PRP'
    MD -> 'MD'
    """
    
    try:
        # Create a grammar that accepts any word in its POS category
        grammar_lines = ["S -> NP VP | VP NP | S CC S | NP VP PP | VP PP"]
        grammar_lines.append("NP -> DT NN | DT JJ NN | NN | NNS | NNP | NNPS | PRP | NP CC NP | CD NN")
        grammar_lines.append("VP -> VB | VBZ | VBP | VBD | VBG | VBN | VP NP | VP PP | MD VP | TO VP")
        grammar_lines.append("PP -> IN NP | TO NP")
        grammar_lines.append("ADJP -> JJ | JJ CC JJ")
        grammar_lines.append("ADVP -> RB")
        
        # Add generic rules for each POS tag
        for token, (word, pos) in zip(non_punct_tokens, non_punct_tagged):
            if pos in ['NN', 'NNS', 'NNP', 'NNPS']:
                grammar_lines.append(f"NN -> '{word}'")
            elif pos in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
                grammar_lines.append(f"VB -> '{word}'")
            elif pos in ['JJ', 'JJR', 'JJS']:
                grammar_lines.append(f"JJ -> '{word}'")
            elif pos in ['DT', 'PDT']:
                grammar_lines.append(f"DT -> '{word}'")
            elif pos == 'IN':
                grammar_lines.append(f"IN -> '{word}'")
            elif pos == 'TO':
                grammar_lines.append(f"TO -> '{word}'")
            elif pos == 'CC':
                grammar_lines.append(f"CC -> '{word}'")
            elif pos in ['PRP', 'PRP$']:
                grammar_lines.append(f"PRP -> '{word}'")
            elif pos == 'MD':
                grammar_lines.append(f"MD -> '{word}'")
            elif pos == 'RB':
                grammar_lines.append(f"RB -> '{word}'")
            elif pos == 'CD':
                grammar_lines.append(f"CD -> '{word}'")
        
        simplified_grammar = "\n".join(grammar_lines)
        parser = nltk.ChartParser(nltk.CFG.fromstring(simplified_grammar))
        trees = list(parser.parse(non_punct_tokens))
        return trees, tagged
    except Exception as e:
        st.write(f"Simplified parsing also failed: {e}")
        return None, tagged

def spacy_parse(sentence, nlp=None):
    if nlp is None:
        nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)
    return doc

def draw_parse_tree(sentence):
    """Draw parse tree using NLTK and display in Streamlit"""
    try:
        # Clean the sentence
        cleaned_sentence = clean_sentence(sentence)
        
        # Get POS tags
        tokens = word_tokenize(cleaned_sentence)
        tagged = pos_tag(tokens)
        
        # Define chunk grammar for tree drawing (similar to your example)
        chunker = RegexpParser("""
            NP: {<DT>?<JJ>*<NN>?<NN>} #To extract Noun Phrases
            P: {<IN>} #To extract Prepositions
            V: {<V.*>} #To extract Verbs
            PP: {<P> <NP>} #To extract Prepositional Phrases
            VP: {<V> <NP|PP>*} #To extract Verb Phrases
        """)
        
        # Parse the sentence
        output = chunker.parse(tagged)
        
        st.write("**Parse Tree Structure:**")
        st.code(str(output))
        
        # Create a matplotlib figure for the tree
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')
        
        # Draw the tree
        output.draw()
        
        # Save the tree to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
        plt.close(fig)
        
        # Display in Streamlit
        buf.seek(0)
        img = Image.open(buf)
        st.image(img, caption="Parse Tree Visualization", use_column_width=True)
        
        return output
        
    except Exception as e:
        st.error(f"Error drawing parse tree: {e}")
        return None

def print_nltk_tree(sentence):
    """Display NLTK parse tree - simplified version without matplotlib"""
    trees, tagged = nltk_parse(sentence)
    
    if trees:
        st.write("**NLTK Parse Tree:**")
        for i, tree in enumerate(trees[:1]):  # Show only first parse
            st.write(f"**Parse {i+1}:**")
            tree_str = str(tree)
            st.code(tree_str)
            
            # Try to show pretty print text version
            try:
                st.write("**Tree Structure (formatted):**")
                pretty_text = tree.pformat()
                st.text_area(f"Tree visualization {i+1}:", value=pretty_text, height=200)
            except Exception as e:
                st.write(f"Could not format tree: {e}")
                st.write("**Tree Structure (raw):**")
                st.text(tree_str)
    else:
        # Don't show error message, just use chunk parsing
        try_chunk_parsing(sentence, tagged)

def try_chunk_parsing(sentence, tagged):
    """Alternative parsing using chunk grammar"""
    # Define comprehensive chunk grammar
    chunk_grammar = """
    NP: {<DT|PDT>?<JJ.*>*<NN.*>+}          # Noun phrases
    VP: {<MD>?<VB.*><NP|PP|ADJP|ADVP|SBAR>*<RP>?}  # Verb phrases
    PP: {<IN|TO><NP>}                      # Prepositional phrases
    ADJP: {<JJ.*>+}                        # Adjective phrases
    ADVP: {<RB.*>+}                        # Adverb phrases
    SBAR: {<IN|DT><S>}                     # Subordinate clauses
    """
    
    chunk_parser = RegexpParser(chunk_grammar)
    chunked = chunk_parser.parse(tagged)
    
    st.write("**Chunk Parse Result:**")
    st.code(str(chunked))
    
    # Don't try to draw with matplotlib in Streamlit to avoid empty boxes
    # Just show the text representation which is cleaner
    
    # Optional: Create a simple ASCII tree representation
    try:
        from nltk.tree import Tree as NLTKTree
        if isinstance(chunked, NLTKTree):
            st.write("**Tree Structure:**")
            tree_text = chunked.pformat()
            st.text(tree_text)
    except:
        pass

def print_spacy_tree(sentence):
    doc = spacy_parse(sentence)
    st.write("**spaCy Dependency Parse:**")
    
    # Display dependency relations with descriptions
    deps = []
    for token in doc:
        deps.append(f"{token.text} ({token.pos_}) --{token.dep_}--> {token.head.text} ({token.head.pos_})")
    st.write("\n".join(deps))
    
    # Try to visualize
    try:
        # For better visualization, create HTML
        html = displacy.render(doc, style="dep", page=True)
        st.components.v1.html(html, height=400, scrolling=True)
    except:
        # Fallback to simple display
        dep_table = []
        for token in doc:
            dep_table.append({
                "Token": token.text,
                "POS": token.pos_,
                "Dependency": token.dep_,
                "Head": token.head.text
            })
        import pandas as pd
        st.table(pd.DataFrame(dep_table))

def enhanced_parse_with_visualization(sentence):
    """Enhanced parsing with better visualization"""
    st.subheader("Enhanced Parse Analysis")
    
    # Clean the sentence first
    cleaned_sentence = clean_sentence(sentence)
    st.write(f"**Cleaned sentence:** {cleaned_sentence}")
    
    # Get POS tags
    tokens = word_tokenize(cleaned_sentence)
    tagged = pos_tag(tokens)
    
    # Display POS tags with explanations
    st.write("**POS Tag Analysis:**")
    tag_data = []
    for word, tag in tagged:
        description = POS_DESCRIPTIONS.get(tag, "Unknown")
        tag_data.append({"Word": word, "POS Tag": tag, "Description": description})
    
    import pandas as pd
    st.table(pd.DataFrame(tag_data))
    
    # 0. DRAW PARSE TREE VISUALIZATION (NEW ADDITION)
    st.write("**0. Parse Tree Visualization:**")
    draw_parse_tree(sentence)
    
    # 1. Constituency Parse (NLTK)
    st.write("**1. Constituency Parse (NLTK):**")
    print_nltk_tree(sentence)
    
    # 2. Dependency Parse (spaCy)
    st.write("**2. Dependency Parse (spaCy):**")
    print_spacy_tree(sentence)
    
    # 3. Chunk Parse (Regex)
    st.write("**3. Chunk Parse (Regex):**")
    chunk_grammar = """
    S: {<NP><VP>|<VP><NP>}
    NP: {<DT|PDT|CD|PRP|PRP\$|WP|WP\$>?<JJ.*>*<NN.*>+}
    VP: {<MD>?<VB.*><NP|PP|ADJP|ADVP|SBAR>*} 
    PP: {<IN|TO><NP>}
    ADJP: {<JJ.*>+}
    ADVP: {<RB.*>+}
    SBAR: {<IN|DT|WDT|WP|WP\$><S>}
    """
    
    chunk_parser = RegexpParser(chunk_grammar)
    chunked = chunk_parser.parse(tagged)
    
    st.code(str(chunked))
    
    # Show additional tree representation if available
    try:
        from nltk.tree import Tree as NLTKTree
        if isinstance(chunked, NLTKTree):
            st.write("**Detailed Tree Structure:**")
            tree_text = chunked.pformat()
            st.text_area("Tree visualization:", value=tree_text, height=200)
    except:
        pass

if __name__ == "__main__":
    # Example usage
    test_sentences = [
        "I like Obama.",
        "The quick brown fox jumps over the lazy dog.",
        "She will analyze the data significantly, and he will review it.",
        "John and Mary found new results in the study!"
    ]
    
    for s in test_sentences:
        print(f"\n=== Testing: {s} ===")
        enhanced_parse_with_visualization(s)