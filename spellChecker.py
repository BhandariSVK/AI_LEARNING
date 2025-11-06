# pip insall nltk
# pip install language_tool_python


import nltk
import language_tool_python
from nltk import word_tokenize, pos_tag

# Download Required Res
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Inititalized Grammer Checking Tool
G_CHECKER = language_tool_python.LanguageTool('en-US') # Java Needs TO Run Local Language Model
# G_CHECKER = language_tool_python.LanguageToolPublicAPI('en-US') # Need TO Remote COnnect



def grammerCheck(input):
    print("input Text : ", input)

    # Grammer CHecking
    error_check = G_CHECKER.check(input)
    if not error_check:
        print("NO Gramatical Mistake found.")
    else:
        print(f'Issue Found on : {len(error_check)} words')
        for item in error_check:
            print(item)

    # Tokenize The Sentence For Correction
    tokens = word_tokenize(input)
    # POS Tagging
    pos_tags = pos_tag(tokens)
    # Correct Word Suggest
    CORRECT = G_CHECKER.correct(input)
    print(CORRECT)

grammerCheck("HI my name os souvik bhandari")
        