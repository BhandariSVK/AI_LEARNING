#   ******************        Text Parsing        ******************

# Defination : 
"""
    Text Parsing means analyzing and structuring text, so that machine can understand its grammatical or logical relationships

"""

# Use Case
"""
    1. Question Answering Systems           ::::  Chatbots, Google Assistant, AI tutors.
    2. Machine Translation                  ::::  Google Translate, DeepL, multilingual chatbots.
    3. Information Extraction               :::: automatic knowledge graph creation, database filling, resume data extraction.
    4. Text Summarization                   :::: Automatic news or research paper summarization.
    5. Sentiment Analysis                   ::::  Product review analysis, social media monitoring.
    6. Grammar Checking / Writing Tools             :::: Grammar correction and writing assistance.
    7. Chatbot & Conversational AI Understanding    :::: Travel assistants, voice bots.
    8. Data Analytics & Document Processing         :::: Automated data entry, compliance checking, OCR text understanding.
"""





# < POS used Here >
# DT  >>>>> 	Determiner	        >>>>>   a, an, the
# NN  >>>>> 	Noun (singular)	    >>>>>   cat, mat, book
# VBD >>>>>	    Verb (past tense)   >>>>>	sat, walked, ran
# IN  >>>>> 	Preposition	        >>>>>   on, in, under, over
# JJ  >>>>>     Adjective           >>>>>   big, blue, fast



# ?   >>>>>     Optional (0 or 1)
# *   >>>>>     Zero or more
# +   >>>>>     One or more




# pip install nltk

import nltk
from nltk import word_tokenize,pos_tag, RegexpParser

nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")

# Input
sentence = "The cat sat on the mat"

# Tokenization
tokens = word_tokenize(sentence)
print("Tokens : " , tokens)  # Tokens :  ['The', 'cat', 'sat', 'on', 'the', 'mat']


# Parts Of Speech
pos_tagss = pos_tag(tokens)
print("pos Tags : ", pos_tagss) # pos Tags :  [('The', 'DT'), ('cat', 'NN'), ('sat', 'VBD'), ('on', 'IN'), ('the', 'DT'), ('mat', 'NN')]

# Chunking
grammer = r"""
    NP: {<DT>?<JJ>*<NN>}            # Noun Phrase
    VP: {<VB.*><NP|PP|CLAUSE>+$}    # Verb Phrase
    PP: {<IN><NP>}                  # Prepositional Phrase
"""

parser = RegexpParser(grammer)
tree = parser.parse(pos_tagss)

print(" <<< TREEEEE >>> ")
print(tree) # (S (NP The/DT cat/NN) sat/VBD (PP on/IN (NP the/DT mat/NN)))



# View The Diagram
tree.draw()