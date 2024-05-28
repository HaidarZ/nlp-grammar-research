from rules import correct_grammar

if __name__ == "__main__":
  text = "She don't like to go to the cinema."
  corrected_text = correct_grammar(text)
  print("Original Text: ", text)
  print("Corrected Text:", corrected_text)