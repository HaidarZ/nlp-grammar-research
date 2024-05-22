from rules import correct_grammar

if __name__ == "__main__":
  text = "The man go to the dr. yesterday."
  corrected_text = correct_grammar(text)
  print("Original Text: ", text)
  print("Corrected Text:", corrected_text)
  text = "dr. smith and mr. jones are going to the market. the cats chases the mouse. a apple is on the tables. he go to school yesterday."
  corrected_text = correct_grammar(text)
  print("Original Text: ", text)
  print("Corrected Text:", corrected_text)