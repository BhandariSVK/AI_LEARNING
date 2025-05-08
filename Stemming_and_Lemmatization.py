words = [
    "eating", "driving", "playing", "sit", "drive", "driving", "gamer", "best", "better", "congratulations", "linear", "souvik", "bhandari",
    "ayan", "DJ", "sound", "system", "water", "worst", "forest", "chain", "tallest", "darkest", "tress", "sentimental", "analysis", "chatrs",
    "portal", "entry", "history", "programing", "programmer", "programs", "prons", 
]



##########################################################  Stemming -> 1. PorterStemer #############################################################################
# It Uses Porter Stemming ALgoritham
from nltk.stem import PorterStemmer
stemming = PorterStemmer()
for word in words:
    stemed = word + " ---------------- > " + stemming.stem(word)
    # print(stemed)

#! :::: Output :::::

# eating ---------------- > eat
# driving ---------------- > drive
# playing ---------------- > play
# sit ---------------- > sit
# drive ---------------- > drive
# driving ---------------- > drive
# gamer ---------------- > gamer
# best ---------------- > best
# better ---------------- > better
# congratulations ---------------- > congratul (need to see)
# linear ---------------- > linear
# souvik ---------------- > souvik
# bhandari ---------------- > bhandari
# ayan ---------------- > ayan
# DJ ---------------- > dj
# sound ---------------- > sound
# system ---------------- > system
# water ---------------- > water
# worst ---------------- > worst
# forest ---------------- > forest
# chain ---------------- > chain
# tallest ---------------- > tallest
# darkest ---------------- > darkest
# tress ---------------- > tress
# sentimental ---------------- > sentiment
# analysis ---------------- > analysi (need to see)
# chatrs ---------------- > chatr (need to see)
# portal ---------------- > portal
# entry ---------------- > entri (need to see)
# history ---------------- > histori (need to see)
# programing ---------------- > program
# programmer ---------------- > programm (need to see)
# programs ---------------- > program
# prons ---------------- > pron (need to see)





########################################################## Stemming -> 2. LancasterStemer #############################################################################
# It Uses Lancaster stemming Algorithm 



from nltk.stem import LancasterStemmer
lancaster = LancasterStemmer()
for word in words:
    stemed = word + " ---------------- > " + lancaster.stem(word)
    # print(stemed)

#! ::: OutPut :::

# eating ---------------- > eat
# driving ---------------- > driv (need to see)
# playing ---------------- > play
# sit ---------------- > sit
# drive ---------------- > driv (need to see)
# driving ---------------- > driv (need to see)
# gamer ---------------- > gam (need to see)
# best ---------------- > best
# better ---------------- > bet (need to see)
# congratulations ---------------- > congrat (need to see)
# linear ---------------- > linear
# souvik ---------------- > souvik
# bhandari ---------------- > bhandar
# ayan ---------------- > ay (need to see)
# DJ ---------------- > dj
# sound ---------------- > sound
# system ---------------- > system
# water ---------------- > wat (need to see)
# worst ---------------- > worst
# forest ---------------- > forest
# chain ---------------- > chain
# tallest ---------------- > tallest
# darkest ---------------- > darkest
# tress ---------------- > tress
# sentimental ---------------- > senty
# analysis ---------------- > analys (need to see)
# chatrs ---------------- > chat (need to see)
# portal ---------------- > port (need to see)
# entry ---------------- > entry
# history ---------------- > hist
# programing ---------------- > program
# programmer ---------------- > program
# programs ---------------- > program
# prons ---------------- > pron (need to see)






########################################################## Stemming -> 3. RegexpStemmer #############################################################################


from nltk.stem import RegexpStemmer
regexstemer = RegexpStemmer('ing|s$|e$|able$', min=5)
for word in words:
    stemed = word + " ---------------- > " + regexstemer.stem(word)
    # print(stemed)

#! ::: OutPut :::

# eating ---------------- > eat
# driving ---------------- > driv
# playing ---------------- > play
# sit ---------------- > sit
# drive ---------------- > driv
# driving ---------------- > driv
# gamer ---------------- > gamer
# best ---------------- > best
# better ---------------- > better
# congratulations ---------------- > congratulation
# linear ---------------- > linear
# souvik ---------------- > souvik
# bhandari ---------------- > bhandari
# ayan ---------------- > ayan
# DJ ---------------- > DJ
# sound ---------------- > sound
# system ---------------- > system
# water ---------------- > water
# worst ---------------- > worst
# forest ---------------- > forest
# chain ---------------- > chain
# tallest ---------------- > tallest
# darkest ---------------- > darkest
# tress ---------------- > tres
# sentimental ---------------- > sentimental
# analysis ---------------- > analysi
# chatrs ---------------- > chatr
# portal ---------------- > portal
# entry ---------------- > entry
# history ---------------- > history
# programing ---------------- > program
# programmer ---------------- > programmer
# programs ---------------- > program
# prons ---------------- > pron








########################################################## Stemming -> 4. SnowballStemmer #############################################################################


from nltk.stem import SnowballStemmer
snowballstemer = SnowballStemmer('english', ignore_stopwords=False)
for word in words:
    stemed = word + " ---------------- > " + snowballstemer.stem(word)
    # print(stemed)

#! ::: OutPut :::

# eating ---------------- > eat
# driving ---------------- > drive
# playing ---------------- > play
# sit ---------------- > sit
# drive ---------------- > drive
# driving ---------------- > drive
# gamer ---------------- > gamer
# best ---------------- > best
# better ---------------- > better
# congratulations ---------------- > congratul (Need To See)
# linear ---------------- > linear
# souvik ---------------- > souvik
# bhandari ---------------- > bhandari
# ayan ---------------- > ayan
# DJ ---------------- > dj
# sound ---------------- > sound
# system ---------------- > system
# water ---------------- > water
# worst ---------------- > worst
# forest ---------------- > forest
# chain ---------------- > chain
# tallest ---------------- > tallest
# darkest ---------------- > darkest
# tress ---------------- > tress
# sentimental ---------------- > sentiment
# analysis ---------------- > analysi
# chatrs ---------------- > chatr
# portal ---------------- > portal
# entry ---------------- > entri
# history ---------------- > histori
# programing ---------------- > program
# programmer ---------------- > programm
# programs ---------------- > program
# prons ---------------- > pron





########################################################## Stemming -> 5. WordNetLemmatizer #############################################################################


# import nltk;  nltk.download('wordnet') # Need This For FirstTime Only

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
"""
    ------ POS TAG ------
    Noun = n
    Verb = v
    Adjective = a
    Adverb = r  
"""

for word in words:
    lemmatized = word + " ---------------- > " + lemmatizer.lemmatize(word, pos='v')
    # print(lemmatized)

#! ::: OutPut :::

# eating ---------------- > eat
# driving ---------------- > drive
# playing ---------------- > play
# sit ---------------- > sit
# drive ---------------- > drive
# driving ---------------- > drive
# gamer ---------------- > gamer
# best ---------------- > best
# better ---------------- > better
# congratulations ---------------- > congratulations
# linear ---------------- > linear
# souvik ---------------- > souvik
# bhandari ---------------- > bhandari
# ayan ---------------- > ayan
# DJ ---------------- > DJ
# sound ---------------- > sound
# system ---------------- > system
# water ---------------- > water
# worst ---------------- > worst
# forest ---------------- > forest
# chain ---------------- > chain
# tallest ---------------- > tallest
# darkest ---------------- > darkest
# tress ---------------- > tress
# sentimental ---------------- > sentimental
# analysis ---------------- > analysis
# chatrs ---------------- > chatrs
# portal ---------------- > portal
# entry ---------------- > entry
# history ---------------- > history
# programing ---------------- > program
# programmer ---------------- > programmer
# programs ---------------- > program
# prons ---------------- > prons











"""
Use Cases :
    1. Sentiment Analysis = Stemming
    2. ChatBots = Lemmatization 
"""