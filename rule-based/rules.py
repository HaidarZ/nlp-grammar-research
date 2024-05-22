import re
import nltk
import spacy
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet, stopwords

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')

# Initialize spaCy
nlp = spacy.load('en_core_web_lg')

# List of common abbreviations
ABBREVIATIONS = {
    'dr.': 'Dr.', 'mr.': 'Mr.', 'mrs.': 'Mrs.', 'ms.': 'Ms.', 'prof.': 'Prof.',
    'i.e.': 'i.e.', 'e.g.': 'e.g.', 'etc.': 'etc.', 'vs.': 'vs.', 'jr.': 'Jr.',
    'sr.': 'Sr.', 'gov.': 'Gov.', 'lt.': 'Lt.', 'sgt.': 'Sgt.', 'col.': 'Col.',
    'gen.': 'Gen.', 'rep.': 'Rep.', 'sen.': 'Sen.', 'rev.': 'Rev.', 'a.m.': 'a.m.',
    'p.m.': 'p.m.', 'b.c.': 'B.C.', 'a.d.': 'A.D.', 'st.': 'St.', 'ave.': 'Ave.',
    'blvd.': 'Blvd.', 'rd.': 'Rd.', 'dr.': 'Dr.', 'mt.': 'Mt.', 'ft.': 'Ft.'
}

# Set of English stopwords
STOPWORDS = set(stopwords.words('english'))

# List of common irregular verbs
IRREGULAR_VERBS = {
    'be': ['am', 'is', 'are', 'was', 'were', 'been', 'being'],
    'beat': ['beat', 'beaten'],
    'become': ['became', 'become'],
    'begin': ['began', 'begun'],
    'bend': ['bent'],
    'bet': ['bet'],
    'bid': ['bid'],
    'bind': ['bound'],
    'bite': ['bit', 'bitten'],
    'bleed': ['bled'],
    'blow': ['blew', 'blown'],
    'break': ['broke', 'broken'],
    'bring': ['brought'],
    'build': ['built'],
    'burn': ['burnt', 'burned'],
    'burst': ['burst'],
    'buy': ['bought'],
    'catch': ['caught'],
    'choose': ['chose', 'chosen'],
    'come': ['came', 'come'],
    'cost': ['cost'],
    'cut': ['cut'],
    'deal': ['dealt'],
    'dig': ['dug'],
    'do': ['did', 'done'],
    'draw': ['drew', 'drawn'],
    'dream': ['dreamt', 'dreamed'],
    'drink': ['drank', 'drunk'],
    'drive': ['drove', 'driven'],
    'eat': ['ate', 'eaten'],
    'fall': ['fell', 'fallen'],
    'feed': ['fed'],
    'feel': ['felt'],
    'fight': ['fought'],
    'find': ['found'],
    'fly': ['flew', 'flown'],
    'forget': ['forgot', 'forgotten'],
    'forgive': ['forgave', 'forgiven'],
    'freeze': ['froze', 'frozen'],
    'get': ['got', 'gotten'],
    'give': ['gave', 'given'],
    'go': ['went', 'gone'],
    'grow': ['grew', 'grown'],
    'hang': ['hung'],
    'have': ['had'],
    'hear': ['heard'],
    'hide': ['hid', 'hidden'],
    'hit': ['hit'],
    'hold': ['held'],
    'hurt': ['hurt'],
    'keep': ['kept'],
    'know': ['knew', 'known'],
    'lay': ['laid'],
    'lead': ['led'],
    'leave': ['left'],
    'lend': ['lent'],
    'let': ['let'],
    'lie': ['lay', 'lain'],
    'light': ['lit', 'lighted'],
    'lose': ['lost'],
    'make': ['made'],
    'mean': ['meant'],
    'meet': ['met'],
    'pay': ['paid'],
    'put': ['put'],
    'read': ['read'],
    'ride': ['rode', 'ridden'],
    'ring': ['rang', 'rung'],
    'rise': ['rose', 'risen'],
    'run': ['ran', 'run'],
    'say': ['said'],
    'see': ['saw', 'seen'],
    'sell': ['sold'],
    'send': ['sent'],
    'set': ['set'],
    'shake': ['shook', 'shaken'],
    'shine': ['shone'],
    'shoot': ['shot'],
    'show': ['showed', 'shown'],
    'shut': ['shut'],
    'sing': ['sang', 'sung'],
    'sink': ['sank', 'sunk'],
    'sit': ['sat'],
    'sleep': ['slept'],
    'speak': ['spoke', 'spoken'],
    'spend': ['spent'],
    'stand': ['stood'],
    'steal': ['stole', 'stolen'],
    'stick': ['stuck'],
    'strike': ['struck'],
    'swear': ['swore', 'sworn'],
    'swim': ['swam', 'swum'],
    'take': ['took', 'taken'],
    'teach': ['taught'],
    'tear': ['tore', 'torn'],
    'tell': ['told'],
    'think': ['thought'],
    'throw': ['threw', 'thrown'],
    'understand': ['understood'],
    'wake': ['woke', 'woken'],
    'wear': ['wore', 'worn'],
    'win': ['won'],
    'write': ['wrote', 'written']
}

# Helper function to get the part of speech tag for lemmatization
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

# Function to apply grammar rules
def correct_subject_verb_agreement(tagged_tokens):
    corrected_tokens = []
    for i, (word, tag) in enumerate(tagged_tokens):
        if tag in ['VBP', 'VBZ'] and i > 0:
            prev_word, prev_tag = tagged_tokens[i-1]
            if prev_tag in ['NN', 'PRP'] and tag == 'VBP':  # Singular noun or pronoun with plural verb
                # Verbs ending in 'o', 'ch', 's', 'sh', 'x', or 'z' should end with 'es'
                if word.endswith(('o', 'ch', 's', 'sh', 'x', 'z')):
                    corrected_tokens.append((word + 'es', 'VBZ'))
                else:
                    corrected_tokens.append((word + 's', 'VBZ'))
            elif prev_tag == 'NNS' and tag == 'VBZ':  # Plural noun with singular verb
                if word.rstrip('s').endswith(('o', 'ch', 's', 'sh', 'x', 'z')):
                    corrected_tokens.append((word.rstrip('es'), 'VBP'))
                else:
                    corrected_tokens.append((word.rstrip('s'), 'VBP'))
            else:
                corrected_tokens.append((word, tag))
        elif tag == 'VBD' and word in IRREGULAR_VERBS:
            base_form = wordnet.morphy(word, wordnet.VERB)
            if base_form and base_form in IRREGULAR_VERBS:
                corrected_tokens.append((IRREGULAR_VERBS[base_form][1], 'VBN'))
            else:
                corrected_tokens.append((word, tag))
        else:
            corrected_tokens.append((word, tag))
    return corrected_tokens

# Function to correct singular/plural noun forms
def correct_noun_forms(tagged_tokens):
    corrected_tokens = []
    for word, tag in tagged_tokens:
        if tag == 'NNS' and not word.endswith('s'):
            singular_form = wordnet.morphy(word, wordnet.NOUN)
            if singular_form and singular_form != word:
                corrected_tokens.append((singular_form + 's', 'NNS'))
            else:
                corrected_tokens.append((word, tag))
        elif tag == 'NN' and word.endswith('s'):
            singular_form = word[:-1]
            if wordnet.synsets(singular_form):
                corrected_tokens.append((singular_form, 'NN'))
            else:
                corrected_tokens.append((word, tag))
        else:
            corrected_tokens.append((word, tag))
    return corrected_tokens

# Function to correct irregular verbs
def correct_irregular_verbs(tagged_tokens):
    corrected_tokens = []
    for i, (word, tag) in enumerate(tagged_tokens):
        if tag.startswith('VB') and word in IRREGULAR_VERBS:
            base_form = wordnet.morphy(word, wordnet.VERB)
            if base_form and base_form in IRREGULAR_VERBS:
                corrected_form = IRREGULAR_VERBS[base_form][0]
                corrected_tokens.append((corrected_form, tag))
            else:
                corrected_tokens.append((word, tag))
        else:
            corrected_tokens.append((word, tag))
    return corrected_tokens

# Function to correct capitalization
def correct_capitalization(sentences):
    corrected_sentences = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        words[0] = words[0].capitalize()  # Capitalize the first word of each sentence
        corrected_sentences.append(' '.join(words))
    return corrected_sentences

# Function to correct punctuation
def correct_punctuation(sentences):
    corrected_sentences = []
    for sentence in sentences:
        sentence = re.sub(r'\s([?.!"](?:\s|$))', r'\1', sentence)  # Remove space before punctuation
        corrected_sentences.append(sentence)
    return corrected_sentences

# Function to correct abbreviations
def correct_abbreviations(text):
    words = word_tokenize(text)
    corrected_words = []
    for word in words:
        lower_word = word.lower()
        if lower_word in ABBREVIATIONS:
            corrected_words.append(ABBREVIATIONS[lower_word])
        else:
            corrected_words.append(word)
    return ' '.join(corrected_words)

# Function to ensure stopwords are not altered
def ensure_stopwords(text):
    words = word_tokenize(text)
    return ' '.join([word for word in words if word.lower() in STOPWORDS or word.lower() not in STOPWORDS])

# Function to correct proper nouns and named entities using spaCy
def correct_named_entities(text):
    doc = nlp(text)
    corrected_tokens = []
    for token in doc:
        if token.ent_type_ in ['PERSON', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT']:
            corrected_tokens.append(token.text.capitalize())
        else:
            corrected_tokens.append(token.text)
    return ' '.join(corrected_tokens)

# Function to correct article usage
def correct_articles(text):
    words = word_tokenize(text)
    corrected_words = []
    for i, word in enumerate(words):
        if word.lower() in ['a', 'an'] and i < len(words) - 1:
            next_word = words[i + 1]
            if next_word[0].lower() in 'aeiou' and word.lower() == 'a':
                corrected_words.append('an')
            elif next_word[0].lower() not in 'aeiou' and word.lower() == 'an':
                corrected_words.append('a')
            else:
                corrected_words.append(word)
        else:
            corrected_words.append(word)
    return ' '.join(corrected_words)

# Main function to correct grammar
def correct_grammar(text):
    # Correct abbreviations first
    text = correct_abbreviations(text)
    
    # Ensure stopwords are correctly used
    text = ensure_stopwords(text)
    
    # Correct named entities
    text = correct_named_entities(text)
    
    # Correct articles
    text = correct_articles(text)
    
    sentences = sent_tokenize(text)
    corrected_text = []
    
    for sentence in sentences:
        words = word_tokenize(sentence)
        tagged = pos_tag(words)
        corrected_tags = correct_subject_verb_agreement(tagged)
        corrected_tags = correct_noun_forms(corrected_tags)
        corrected_tags = correct_irregular_verbs(corrected_tags)
        
        corrected_sentence = ' '.join([word for word, tag in corrected_tags])
        corrected_text.append(corrected_sentence)
    
    # Apply capitalization correction
    corrected_text = correct_capitalization(corrected_text)
    
    # Apply punctuation correction
    corrected_text = correct_punctuation(corrected_text)
    
    return ' '.join(corrected_text)
