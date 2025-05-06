#** Libs 
#! pip install nltk
#**


################################################################ Need Only For 1st Time Run ################################################################
#import nltk
# nltk.download('punkt_tab') 
################################################################ Need Only For 1st Time Run ################################################################




################################################# CORPUS (START) ###################################################################################################
corpus = """
        Hi My Name is Souvik Bhandari. I am from Bankura. Not Proper Bankura, my village is actually located near joypur. 
        And After Hearing This you will say, Oh! my God, Jungle. Yes Their is a jungle near my house.
        If you ask me, is their any resort near your home? I definately Say yes I have. But i don't love that so much.
        When I was child I am seeing that from scrach, when that was under construction, and actually the moto of that
        resort is to give feel about village culture to travellers. But Luckily I am a villager, that why i dont have
        so much interest on that restort.
"""
################################################# CORPUS (START) ####################################################################################################



################################################# USING : Sentance Tokenizer : sent_tokenize (START) ################################################################
from nltk.tokenize import sent_tokenize
resp = sent_tokenize(corpus)
# print(resp)

#! :::: OUTPUT :::
[
    '\n        Hi My Name is Souvik Bhandari.', 
    'I am from Bankura.', 
    'Not Proper Bankura, my village is actually located near joypur.', 
    'And After Hearing This you will say, Oh!', 
    'my God, Jungle.', 
    'Yes Their is a jungle near my house.', 
    'If you ask me, is their any resort near your home?', 
    'I definately Say yes I have.', 
    "But i don't love that so much.", 
    'When I was child I am seeing that from scrach, when that was under construction, and actually the moto of that\n        resort is to give feel about village culture to travellers.', 
    'But Luckily I am a villager, that why i dont have\n        so much interest on that restort.'
]
################################################# USING : Sentance Tokenizer : sent_tokenize (END) ##################################################################




################################################# USING : Word Tokenizer : word_tokenize (START) ###################################################################
from nltk.tokenize import word_tokenize
resp = word_tokenize(corpus)
# print(resp)

#! :::: OUTPUT :::
[
    'Hi', 'My', 'Name', 'is', 'Souvik', 'Bhandari', '.', 'I', 'am', 'from', 'Bankura', '.', 'Not', 'Proper', 'Bankura', ',', 'my', 'village', 'is', 'actually', 
    'located', 'near', 'joypur', '.', 'And', 'After', 'Hearing', 'This', 'you', 'will', 'say', ',', 'Oh', '!', 'my', 'God', ',', 'Jungle', '.', 'Yes', 'Their', 
    'is', 'a', 'jungle', 'near', 'my', 'house', '.', 'If', 'you', 'ask', 'me', ',', 'is', 'their', 'any', 'resort', 'near', 'your', 'home', '?', 'I', 'definately', 
    'Say', 'yes', 'I', 'have', '.', 'But', 'i', 'do', "n't", 'love', 'that', 'so', 'much', '.', 'When', 'I', 'was', 'child', 'I', 'am', 'seeing', 'that', 'from', 
    'scrach', ',', 'when', 'that', 'was', 'under', 'construction', ',', 'and', 'actually', 'the', 'moto', 'of', 'that', 'resort', 'is', 'to', 'give', 'feel', 'about',
      'village', 'culture', 'to', 'travellers', '.', 'But', 'Luckily', 'I', 'am', 'a', 'villager', ',', 'that', 'why', 'i', 'dont', 'have', 'so', 'much', 'interest', 
      'on', 'that', 'restort', '.'
]

################################################# USING : Word Tokenizer : word_tokenize (END) ##################################################################







################################################# USING : Word Tokenizer : wordpunct_tokenize (START) ###################################################################
from nltk.tokenize import wordpunct_tokenize
resp = wordpunct_tokenize(corpus)
# print(resp)

#! :::: OUTPUT :::
[
    'Hi', 'My', 'Name', 'is', 'Souvik', 'Bhandari', '.', 'I', 'am', 'from', 'Bankura', '.', 'Not', 'Proper', 'Bankura', ',', 'my', 'village', 'is', 'actually', 
    'located', 'near', 'joypur', '.', 'And', 'After', 'Hearing', 'This', 'you', 'will', 'say', ',', 'Oh', '!', 'my', 'God', ',', 'Jungle', '.', 'Yes', 'Their', 
    'is', 'a', 'jungle', 'near', 'my', 'house', '.', 'If', 'you', 'ask', 'me', ',', 'is', 'their', 'any', 'resort', 'near', 'your', 'home', '?', 'I', 'definately',
      'Say', 'yes', 'I', 'have', '.', 'But', 'i', 'don', "'", 't', 'love', 'that', 'so', 'much', '.', 'When', 'I', 'was', 'child', 'I', 'am', 'seeing', 'that', 
      'from', 'scrach', ',', 'when', 'that', 'was', 'under', 'construction', ',', 'and', 'actually', 'the', 'moto', 'of', 'that', 'resort', 'is', 'to', 'give', 
      'feel', 'about', 'village', 'culture', 'to', 'travellers', '.', 'But', 'Luckily', 'I', 'am', 'a', 'villager', ',', 'that', 'why', 'i', 'dont', 'have', 'so',
        'much', 'interest', 'on', 'that', 'restort', '.'
]
# Differnece between word_tokenize and wordpunct_tokenize is     <<< 'do', "n't" >>>          <<<'don', "'", 't'>>>
# This Method Tokenize Word at its puntuation level also

################################################# USING : Word Tokenizer : wordpunct_tokenize (END) ##################################################################