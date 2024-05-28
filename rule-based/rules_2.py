import spacy
import nltk
from nltk.corpus import wordnet as wn

# Load spaCy model
nlp = spacy.load('en_core_web_lg')

# Ensure NLTK WordNet corpus is downloaded
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


def get_singular_verb_form(verb):
    """Get the third person singular present form of a verb."""
    if verb.endswith(('o', 'ch', 's', 'sh', 'x', 'z')):
        return verb + 'es'
    else:
        return verb + 's'


def correct_subject_verb_agreement(doc):
    corrections = []
    contractions = {"do": "doesn't", "don't": "doesn't", "doesn't": "do", "have": "hasn't", "hasn't": "have"}

    for token in doc:
        if token.dep_ == 'nsubj' and token.head.pos_ == 'VERB':
            subject = token
            verb = token.head

            if subject.tag_ in ['NN', 'NNP', 'PRP'] and verb.tag_ in ['VB', 'VBP']:
                if subject.text.lower() in ["he", "she", "it"] and verb.text == "do":
                    correct_form = "does"
                else:
                    correct_form = get_singular_verb_form(verb.lemma_)
                if verb.text != correct_form:
                    corrections.append((verb, correct_form))
            elif subject.tag_ in ['NNS', 'NNPS'] and verb.tag_ == 'VBZ':
                correct_form = verb.lemma_
                if verb.text != correct_form:
                    corrections.append((verb, correct_form))
            elif subject.text.lower() in ["he", "she", "it"] and verb.text in contractions:
                correct_form = contractions[verb.text]
                if verb.text != correct_form:
                    corrections.append((verb, correct_form))
    return corrections


def correct_articles(doc):
    corrections = []
    vowels = 'aeiou'
    for token in doc:
        if token.pos_ == 'DET':
            next_token = token.nbor(1) if token.i + 1 < len(doc) else None
            if next_token and next_token.pos_ == 'NOUN':
                if token.text.lower() == 'a' and next_token.text[0].lower() in vowels:
                    corrections.append((token, 'an'))
                elif token.text.lower() == 'an' and next_token.text[0].lower() not in vowels:
                    corrections.append((token, 'a'))
    return corrections



def correct_proper_noun_capitalization(doc):
    corrections = []
    for token in doc:
        if token.pos_ == 'PROPN' and not token.text[0].isupper():
            corrections.append((token, token.text.capitalize()))
    return corrections


def apply_corrections(sentence, corrections):
    for token, correction in corrections:
        sentence = sentence[:token.idx] + correction + sentence[token.idx + len(token.text):]
    return sentence


def grammar_correction_pipeline(sentence):
    doc = nlp(sentence)

    corrections = []
    corrections.extend(correct_subject_verb_agreement(doc))
    corrections.extend(correct_articles(doc))
    corrections.extend(correct_proper_noun_capitalization(doc))

    corrected_sentence = apply_corrections(sentence, corrections)

    return corrected_sentence, corrections


if __name__ == "__main__":
    # # Example usage:
    # sentence = "The dogs barks in a park."
    # corrected_sentence, corrections = grammar_correction_pipeline(sentence)
    # print("Corrected Sentence:", corrected_sentence)
    # for correction in corrections:
    #     print(correction)
    while True:
        input_text = input("Please enter the text to be corrected (or type 'exit' to quit): ")
        if input_text.lower() == 'exit':
            break

        corrected_sentence, corrections = grammar_correction_pipeline(input_text)

        print(f"\nCorrected text:\n{corrected_sentence}")
        print(f"Corrections: {corrections}\n")
