from __future__ import absolute_import, division, print_function 

import numpy 
import spacy
import json
import random
import xang
import sys
import nltk 
import re 
import wikipedia
import html2text
import datetime
import airtable
import multiprocessing 
from random import randint 
from rake_nltk import Rake
from word2number import w2n
from bs4 import BeautifulSoup
from nltk.corpus import wordnet
from num2words import num2words
from dateutil.parser import parse 
from nltk.stem import PorterStemmer
import gensim.models.word2vec as w2v 
from urllib.request import Request, urlopen
from spacy.lang.en.stop_words import STOP_WORDS
from pytextrank import json_iter, limit_keyphrases, limit_sentences, make_sentence, normalize_key_phrases, parse_doc, pretty_print, rank_kernel, render_ranks, text_rank, top_sentences

class GenerateSentences(): # The methods ParseText, RankedGraph and ProduceSentences are adapted from the pytextRank API
    def __init__(self,Text_PhraseLimit, sentWordLimit):
        self.spacy_nlp = None
        self.skip_ner = True
        self.phrase_limit = Text_PhraseLimit
        self.stopwords = STOP_WORDS
        self.sentenceWordLimit = sentWordLimit # Sentence word limit

    def text2json(self,text): # Check name
        json={"id":"1","text":text}
        yield(json)

    def ParseText(self,text): # Parse Text and Convert to Json
        parse=parse_doc(self.text2json(text))
        parse_list=[json.loads(pretty_print(i._asdict())) for i in parse]
        self.parse_list = parse_list

        return parse_list

    def RankedGraph(self, parse_list):
        graph, ranks = xang.text_rank(parse_list)
        norm_rank=xang.normalize_key_phrases(parse_list, ranks, self.stopwords, self.spacy_nlp, self.skip_ner)
        norm_rank_list=[json.loads(pretty_print(rl._asdict())) for rl in norm_rank ]
        phrases = ", ".join(set([p for p in xang.limit_keyphrases(norm_rank_list, self.phrase_limit)]))

        # return a matrix like result for the top keywords
        kernel = xang.rank_kernel(norm_rank_list)
        self.phrases = phrases
        self.kernel = kernel

        return kernel

    def ProduceSentences(self):
        top_sent=xang.top_sentences(self.kernel, self.parse_list)
        top_sent_list=[json.loads(pretty_print(s._asdict())) for s in top_sent]
        sent_iter = sorted(xang.limit_sentences(top_sent_list, self.sentenceWordLimit), key=lambda x: x[1])

        return sent_iter

    def RankSentences(self, sent_iter): # Return ranked sentences
        s=[]
        for sent_text, idx in sent_iter:
            sentence = xang.make_sentence(sent_text)
            SenContents = sentence.split(" ")
            if len(SenContents) >= 7:
                s.append(sentence)
        return s

    def GetPhrases(self):
        return self.phrases

class SelectWebsites():
    def identifyTopic(self,topic): # These are the possible topics they can play
        topic = topic.lower()
        self.availableTopics = self.possibleTopics()
        self.topicId = -1 # An error Flag (Topic does not exist)

        if(topic == "history"):
            self.topicId = 0
        elif(topic == "english literature"):
            self.topicId = 1
        elif(topic == "space"):
            self.topicId = 2
        elif(topic == "biology"):
            self.topicId = 3
        elif(topic == "einstein"):
            self.topicId = 4
        if(self.topicId >= 0): # The topicId does not reflect an error flag
            webpage = self.selectWebpage()
            return webpage

        return "Invalid Topic"

    def selectWebpage(self):
        if self.topicId == 2 or self.topicId == 3 or self.topicId == 4:
            self.max = 1
        else:
            self.max = 2 # Max is the maximum cell number for an array in 'possibleTopics' (i.e. 2 in History with an array size of 3)

        randNum = randint(0,self.max)
        topic = self.availableTopics[self.topicId]
        return topic[randNum][0]

        return -1 # Return Flag

    def possibleTopics(self):
        specificTopics = []

        History = [
            ['World War II'], # Wikipedia
            ['Black Death'],  # Wikipedia
            ['http://www.softschools.com/timelines/world_war_ii_timeline/120/']
        ]

        English_Literature = [
            ['Macbeth'], # Wikipedia
            ['Romeo and Juliet'], # Wikipedia
            ['http://www.softschools.com/literature/summary/macbeth/'],
            ['Pride and Prejudice']
        ]

        Space = [
            ['Black hole'], # Wikipedia
            ['Milky Way'] # Wikipedia
        ]

        einstein = [
            ['https://kids.kiddle.co/Albert_Einstein'],
            ['https://www.theschoolrun.com/homework-help/albert-einstein']
        ]

        Biology = [
            ['https://kids.kiddle.co/DNA'],
            ['Heart'] # Wikipedia
        ]
        specificTopics.append(History)
        specificTopics.append(English_Literature)
        specificTopics.append(Space)
        specificTopics.append(Biology)
        specificTopics.append(einstein)
        self.allTopics = specificTopics

        return specificTopics

    def getTopics(self):
        return self.allTopics, self.topicId, self.max

class generateOptions(): # Obtains the MCQ options
    def __init__(self,numOptions):
        self.numOptions = numOptions # The number of multiple choice options

    def answerOptions(self,answer,possibleWords): # Select the top 4 words for each group
        allOptions = [] # Contains all valid options (i.e. those with a different stem to the answer)
        self.answer = answer
        allOptions.append(self.selectOptions(possibleWords))

        return allOptions

    def selectOptions(self,option):# Take the top four most similar words
        word = []
        validOptions = self.checkStemming(option)
        numOptions = len(validOptions)
        iterOptions = 0

        if(numOptions >= self.numOptions): iterOptions = self.numOptions # There are enough options available
        elif(numOptions < self.numOptions):
            return "No Options" # There are not enough MCQ Options

        for x in range(iterOptions):
            word.append(validOptions[x])

        if(len(word) >= 2): # The USER must have at least 3 options for each question (including the answer)
            return word
        else:
            return -1 # A Flag that there are not enough options

    def checkStemming(self,options): # Ensures the stems of the answer and options are different
        index = 0
        porter = PorterStemmer() # PorterStemmer object
        NewOptions = options.copy()

        for choice in options:
            if(isinstance(choice, tuple)):
                choiceStem = porter.stem(choice[0])
            else:
                choiceStem = porter.stem(choice)
            answerStem = porter.stem(self.answer)
            if(choiceStem == answerStem):
                del NewOptions[index]
            index += 1

        return NewOptions

def CleanText(text):
    return text.replace('=','').replace('"','').replace('"','')

def SetUpNLTK():
    nltk.download('punkt')# Pre-trained Tokeniser -- Convert a piece of text into Tokens (words,sentences,characters etc.)
    nltk.download('averaged_perceptron_tagger')
    nltk.download("stopwords")
    nltk.download('wordnet')

def get_similar_words(word):

    synsets = wordnet.synsets(word, pos='n')

    if len(synsets) == 0: # If there aren't any synsets or hypernyms, return an empty list
        return []
    else:
        synset = synsets[0]

    if len(synset.hypernyms()) == 0:
        return []

    hypernym = synset.hypernyms()[0] # Get the hypernym for the synset
    hyponyms = hypernym.hyponyms() # Aquire hypernyms for the new hypernym

    similar_words = []
    for hyponym in hyponyms:
        similar_word = hyponym.lemmas()[0].name().replace('_', ' ')
        if similar_word != word:
            similar_words.append(similar_word)
        if len(similar_words) == 8:
            break

    return similar_words

class controlGame(): # This Plays the Game -- Generates all questions
    def __init__(self):
        self.validTopic = False
        self.website = SelectWebsites()

    def setAlexaQAs(self,AlexaQA):
        self.AlexaQA = AlexaQA

    def startGame(self):
        print("Lets play Academia-Trivia")

        self.user = input("Please state your username or say 'new user' if you are a new player > ")

        if (self.user == 'new user'):
            self.createUser()
        elif (self.searchUser() == False):
            self.createUser()
        else:
            print("Welcome back " + str(self.user) + ", lets play.")

        self.SetUpNLTK() # Ensures the latest version of the Natural Language Toolkit is used
        self.takeInput()

        return self.text

    def takeInput(self):
        self.topic = input("Choose a topic from History,English-Literature,Space or Biology >    ")
        self.webpage = self.website.identifyTopic(self.topic)
        self.recieveTopicInfo()

    def searchUser(self):  # Cloud database via 'AirTable'
        userExists = False
        DB_Table_Name =  "Scores"
        DB_API_Key = 'keyjqhidne9o6LnXW' # Database API Key
        DB_base_key = 'appxT84c1U9S52Ca2' # Database base key

        self.database = airtable.Airtable(DB_base_key, DB_Table_Name, api_key=DB_API_Key)
        self.DBinfo = self.database.get_all() # Recieves information from the database

        for name in self.DBinfo:
            userId = (name['fields']['Name']).lower()
            if(self.user == userId):
                userExists = True

        if userExists: return True # The username exists

        return False # The username does not exist

    def createUser(self):
        self.database.insert({'Name': str(self.user)})

    def recieveTopicInfo(self):
        if self.webpage == "Invalid Topic":
            print("\n\n","Invalid Topic. Please choose again.")
            return self.takeInput()

        if self.webpage.startswith("http") or self.webpage.startswith("https"): # URL Link
            self.parseWebpage()
        else: # Wikipedia search Term
            self.parseWiki()

        return self.text

    def parseWebpage(self): # CHANGE SOME OF THIS
        url = Request(self.webpage, headers={'User-Agent': 'Mozilla/5.0'})

        try:
            html = urlopen(url).read()
        except: # URL has been changed or website removed
            AllWebsites,topicId,max = self.website.getTopics()
            WebsiteArray = AllWebsites[topicId]
            randNum = randint(0,max)
            self.webpage = WebsiteArray[randNum][0]

            return self.recieveTopicInfo()

        soup = BeautifulSoup(html)

        for script in soup(["script", "style"]): script.extract() # Stop script and style elements
        text = soup.body.get_text() # get text

        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        self.text = '\n'.join(chunk for chunk in chunks if chunk)

    def parseWiki(self):
        self.text = wikipedia.page(title = self.webpage).content
        self.text = self.CleanText()

    def CleanText(self):
        return self.text.replace('=','').replace('"','').replace('"','')

    def SetUpNLTK(self):
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download("stopwords")
        nltk.download('wordnet')

    def playGame(self):
        score = 0
        RightWrong = [] # The number of correct responses
        print("Lets play Academia-Trivia")

        for index in range(len(self.AlexaQA)): # Iterate through all questions
            RightWrong.append(self.recieveAnswer(self.AlexaQA[index],index+1)) # Add one or -1 if the answer is correct or incorrect

        for answer in RightWrong: # Add up the number of correct responses
            if answer == 1: score += 1
            elif answer == -1: score -= 1

        print("You scored " + str(score) + " Out of " + str(len(self.AlexaQA)))

    def recieveAnswer(self,dict,QuesNumber):
        index = 0
        sent = []
        optionList = []
        options = ""
        blankStr = False
        question = dict["sentence"]
        [sent.append(charac) for charac in question]

        for char in sent:
            if char == '_' and blankStr == False:
                blankStr = True
                sent[index] = "'BLANK!'"
            index+=1

        if(blankStr == False):
            return -1

        for choice in dict["options"]:
            options = options + (choice+", ")

        sent = ''.join(sent)
        newstr = sent.replace("_", "")

        # Prints the question to be played in the terminal
        print("\n\n")
        print(("Question " + str(QuesNumber)), "\n")
        print("\n", newstr, "\n")
        print("\nYour options are: " + options, "\n")

        Player_Answer = self.repeat(newstr,options)
        if Player_Answer == dict["answer"]:
            print("That is correct")
            return 1
        else:
            print("Incorrect... The correct answer is " + dict["answer"])
            return 0

    def repeat(self, newstr,options):
        Player_Answer = input(">")
        if Player_Answer == "repeat":
            print("\n", newstr, "\n")
            print("\nYour options are: " + options, "\n")
            return self.repeat(newstr,options)
        return Player_Answer

Game = controlGame()
text = Game.startGame() # Cleaned text

stop_words = STOP_WORDS
Text_PhraseLimit = 15
sentWordLimit = 600 # Sentence word limit/Number of sentences

sentence = GenerateSentences(Text_PhraseLimit,sentWordLimit)
parse_list = sentence.ParseText(text)
kernel = sentence.RankedGraph(parse_list)
phrases = sentence.GetPhrases()

sent_iter = sentence.ProduceSentences()
RankedSentences = sentence.RankSentences(sent_iter)
sentences = RankedSentences

# Set up rake with stop_word directory
stop_dir = stop_words
list = list(stop_words)

# Extract Keywords
keywords = []
FinalKeyWords = []
r = Rake() # Uses stopwords for english from NLTK, and all puntuation characters.
content = ''.join(sentences)

def isIllegalNumber(word): # Numbers such as '10,000 ' with commas are illegal in python so a regular expression is used to convert to 10000
        regnumber = re.compile('^(?=.)(\d{1,3}(,\d{3})*)?(\.\d+)?$')
        NewWord = ""

        if regnumber.search(word):
                NewWord = word.replace(',','')
                return True,NewWord
        else:
            return False,word

index = 0
for x in sentences:
    miniSent = x.split()
    subIndex = 0
    sentenceIndex = 0
    for y in miniSent:
        bool,Word = isIllegalNumber(y)
        if bool:
            miniSent[sentenceIndex] = Word
        sent = ' '.join(miniSent)
        sentences[index] = sent
        subIndex+=1
        sentenceIndex+=1
    index+=1

# Take the first one if they all rank the same, then take the noun or adjective
SelectedPhrases = []
keywordsSents = [] # Stores the sentence, its keywords/phrases and an ID

for sentence in sentences:
    text_list = sentence.split()
    phraseTag = nltk.pos_tag(text_list)

    if(not(phraseTag[0][1] == 'RB')):# If a sentence starts with an adverb dont use it because these sentences usually rely on information from the prior sentence
        words = r.extract_keywords_from_text(sentence)
        x = r.get_ranked_phrases()
        y = r.get_ranked_phrases_with_scores()
        SelectedPhrases.append(y)

        wordSent = []
        wordSent.append(y) # Save the phrase
        wordSent.append(sentence) # Save the sentence
        keywordsSents.append(wordSent) # Save the combination

def ContainsNumberTag(SpeechTags):
    for letter,Tag in SpeechTags:
        if(Tag == 'CD'):
            return True
    return False

def IsNumber(KeyWordPhrases):

    for Words in KeyWordPhrases:
        string = Words[1]
        text_list = string.split()
        temp = nltk.pos_tag(text_list)
        if(ContainsNumberTag(temp)):
            return Words[1]

    return "No Number" # No date or number (e.g. of objects)

def LastWordIsNoun(phraseTag):
    if(phraseTag[len(phraseTag)-1][1] == 'NN'):
        return phraseTag[len(phraseTag)-1][0]
    else:
        return 'No Noun'

def SelectWord(phraseTag,text_list): # This function contains the Syntatic Filter
    temp = ' '.join(map(str, text_list))
    NoNouns = 0

    if(ContainsNumberTag(phraseTag)): # If it contains a number (e.g. date) select it
        return temp

    index = 0
    for phrases,Tag in phraseTag:
        if(Tag == 'NN'):
            try:
                Word,tag = phraseTag[index+1]
                if(tag == 'NNS'): # If its a Noun followed by NOUN Plural
                    ChosenWord = phrases + ' ' + Word
                    return ChosenWord
            except:
                NoNouns += 1

            return phrases # Return the Noun only
        index += 1

    for phrases,Tag in phraseTag:
        if(Tag == 'JJ'):
            try:
                Word,tag = phraseTag[index+1]
                if(tag == 'NNS'):
                    ChosenWord = phrases + ' ' + Word
                    return ChosenWord
            except:
                NoNouns += 1
        index += 1

    for phrases,Tag in phraseTag:
        if(Tag == 'NNS'): # Noun Plural
            return phrases

    for phrases,Tag in phraseTag:
        if(Tag == 'VBN'):
            return phrases

    return "No Keywords"

def RemovePhrases(AllPhrases,numPhrase): # Remove all the phrases other than the number
    index = 0
    iterPhrases = AllPhrases.copy()

    for phrase in AllPhrases:
        if(not(phrase[1] == numPhrase)):
            del iterPhrases[index]
            index -= 1
        index += 1


    return iterPhrases[0][1]

index = 0
SelectedKeys  = []
NumsChosen = 0 # limit the number of times we select a number

for AllPhrases in SelectedPhrases:
    phrases = IsNumber(AllPhrases)
    if(not(phrases == "No Number") and NumsChosen <= 3):
        NumsChosen += 1
        SelectedKeys.append(phrases)
        sentPhraseIdGroup = keywordsSents[index]
        keywordsSents[index][0] = RemovePhrases(sentPhraseIdGroup[0],phrases)
    else:
        sentPhraseIdGroup = keywordsSents[index]
        keywordsSents[index][0] = RemovePhrases(sentPhraseIdGroup[0],AllPhrases[0][1])

        PhraseRank = AllPhrases[0] # Take the highest ranking keyword/phrase from the list
        SelectedKeys.append(PhraseRank[1]) # Append the phrases only
    index += 1

ChosenWords = []
for x in SelectedKeys: # Select a keyword from a phrase
    text_list = x.split()
    phraseTag = nltk.pos_tag(text_list)
    chosen = SelectWord(phraseTag,text_list)
    ChosenWords.append(chosen)

index = 0
Chosen = []
for x in keywordsSents:
    text_list = x[0].split()
    phraseTag = nltk.pos_tag(text_list)
    chosen = SelectWord(phraseTag,text_list)
    keywordsSents[index][0] = chosen
    Chosen.append(chosen)
    index += 1

# Convert a sentence to an array of individual words
def ListOfWords(sentence):
    return sentence.split()

# Split corpus into sentences
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# The tokeniser will be used to tokensize the corpus
raw_sentences = tokenizer.tokenize(text) # Split text into sentences
Tokenised_sents = [] # Tokenise each word in a sentence
for sentence in raw_sentences:
    if(len(sentence) > 0):
        ArrayOfWords = ListOfWords(sentence)
        Tokenised_sents.append(ArrayOfWords)
    else:
        print("ERROR")
        exit(2)

index = 0
for x in Tokenised_sents:
    subIndex = 0
    for y in x:
        bool,Word = isIllegalNumber(y)
        if bool: Tokenised_sents[index][subIndex] = Word
        subIndex += 1
    index+= 1

class ManageVectorModel(): # Initialises the Word Vector (Word2Vec) Model
    def __init__(self,Tokenised_sents):
        self.dimentions = 300 # Number of dimentions for the vectors
        self.minWords = 3 # The smallest number of words to be processed when converting to a vector
        self.processing = multiprocessing.cpu_count() # The number of threads to run in parallel. The more we have, the faster the training
        self.CWL = 7 # Context window length
        self.downsampling = 1e-3 # How often we process the same word. The more frequent a word, the less we want to use it to create vectors
        self.seed = 1 # Seeding for the Random Number Generator
        self.Tokenised_sents = Tokenised_sents # Tokenised sentences

    def buildModel(self):
        self.model = w2v.Word2Vec(
            sg=1,
            seed=self.seed,
            workers=self.processing,
            size=self.dimentions,
            min_count=self.minWords,
            window=self.CWL,
            sample=self.downsampling
        )
        self.buildVocabulary() # Build the vocabulary of the model
        self.trainModel() # Train the model
        self.repetitiveTraining() # Allows the model to be trained multiple times

        return self.model

    def buildVocabulary(self): # Build the models vocabulary
        self.model.build_vocab(self.Tokenised_sents)
        self.totExamples = self.model.corpus_count

    def trainModel(self):
        self.model.train(self.Tokenised_sents,total_examples=self.totExamples,epochs=self.model.iter)

    def repetitiveTraining(self):
        self.model.build_vocab(self.Tokenised_sents,update=True)
        self.trainModel()

VM = ManageVectorModel(Tokenised_sents)
vectorModel = VM.buildModel()

#NewModel = VM.LoadModel("VectorModel.w2v") # Load the Model from memory

class InVocab(): # Checks that the words are in the models vocabulary or are dates
    def __init__(self,vocab,ChosenWords):
        self.vocab = vocab
        self.keyWords = ChosenWords

    def IsDate(self,word): # Checks if the keyword is a date
        text_list = word.split()
        wordTag = nltk.pos_tag(text_list)
        return ContainsNumberTag(wordTag)

    def VocabChosen(self,keywordsSents): # Selects keywords that are in the NLP models vocabulary
        index = 0
        selectedWords = []
        selectedNumbers = []
        keywordsSentsCopy = keywordsSents.copy()
        test = keywordsSents.copy()

        for x in keywordsSents:
            inVocab = False
            isDate = False
            for word in self.vocab:
                if(word == x[0]): # If the word is in the models vocabulary
                    selectedWords.append(x[0])
                    inVocab = True
            if(self.IsDate(x[0])): # If its a date
                selectedNumbers.append(x[0])
                isDate = True
            if(inVocab == False and isDate == False): # Its not in the vocab and not a date
                del keywordsSentsCopy[index]
                index -= 1 # Prevents the index from over-running the array
            index += 1

        return (selectedWords,selectedNumbers,keywordsSentsCopy)

    def SelectNumbers(self, numList,keywordsSents): # Select how many numbers / dates for questioning

        count = 0
        datesToTake = 4 # The number of dates to select
        usedNumbers = []
        numList.sort(key = lambda x: len(x),reverse=True)
        keywordsSents = self.isAllNumbers(numList,keywordsSents)

        if(len(keywordsSents) >= 1 and len(keywordsSents) <= datesToTake): # Take three dates
            return keywordsSents

        elif(len(keywordsSents) >= 3):
            newList = []
            while len(usedNumbers) < len(keywordsSents):
                randNum = randint(0,len(keywordsSents)-1)
                if(not(randNum in usedNumbers)):
                    usedNumbers.append(randNum)
                    answer = keywordsSents[randNum][0]
                    if(len(keywordsSents[randNum][0].split()) <= 4 and count <= datesToTake): # Take four dates
                        newList.append(answer)
                        count += 1

            return keywordsSents

        return "No Numbers"

    def isAllNumbers(self,numberList,keywordsSents): # Some words may have incorrect Tags (i.e. archipolago has a CARDINAL number tag but is not a CARDINAL number --- Ensure its a number )
            index = 0
            NoNum = 0
            numbers = numberList.copy()
            numbers2 = keywordsSents.copy()

            for x in keywordsSents:
                isNumber = False
                text_list = x[0].split()
                phraseTag = nltk.pos_tag(text_list)
                for phrase,Tag in phraseTag:
                    if(Tag == 'CD'):
                        try: # Check if its a numerical number (i.e. 1,2,3)
                            num = num2words(phrase)
                            isNumber = True
                        except: # Its not a numerical number
                            try: # Check its a word number (i.e. one,two,three)
                                num = w2n.word_to_num(phrase)
                                isNumber = True
                            except:
                                NoNum += 1
                if(isNumber == False):
                    del numbers2[index]
                    index -= 1
                index += 1

            return numbers2

class TrainVectorModel(): # Re-Training the Word2Vec Model
    def __init__(self,FinalWords,model,tokenizer):
        self.keyWords = FinalWords
        self.vectorModel = model
        self.tokenizer = tokenizer

    def updateModel(self):
        synonyms = []
        for keyword in self.keyWords[0]:
            for syn in wordnet.synsets(keyword):
                for lex in syn.lemmas():
                    synonyms.append(lex.name())

        for x in self.keyWords:
            additionalTerms = wikipedia.search(x)
            for term in additionalTerms:
                termData = self.retrieveWiki(term)
                if(not(termData == "Error")): # The search-term was successful
                    Tokenised_sents = self.tokeniseText(termData)
                    self.trainVectors(Tokenised_sents)

        return self.vectorModel

    def retrieveWiki(self,searchTerm):
        try: # The search terms are dynamic so they may not be adequate for wikipedia
            wikiInfo = wikipedia.page(title = searchTerm).content
            wikiInfo = CleanText(wikiInfo)
            return wikiInfo
        except:
            return "Error"

    def tokeniseText(self,text):
        raw_sentences = self.tokenizer.tokenize(text) # Split text into sentences
        Tokenised_sents = [] # Tokenise each word in a sentence
        for sentence in raw_sentences:
            if(len(sentence) > 0):
                ArrayOfWords = sentence.split()
                Tokenised_sents.append(ArrayOfWords)
            else:
                print("\n\n\n\n\nERROR\n\n\n")
                exit(2)

        return Tokenised_sents

    def trainVectors(self,Tokenised_sents):
        self.vectorModel.build_vocab(Tokenised_sents,update=True) # Update the prior trained model with a new corpus
        self.vectorModel.train(Tokenised_sents,total_examples=self.vectorModel.corpus_count,epochs=self.vectorModel.iter)

class QuestionAnswer(): # This class generates the final sentences,answers and MCQ options
    def __init__(self,numMCQoptions,vectorModel,ChosenWords,keywordsSents,tokenizer):
        self.numMCQoptions = numMCQoptions # Max Four Multiple choice options -- 5 Including the answer
        self.vectorModel = vectorModel
        self.ChosenWords = ChosenWords
        self.keywordsSents = keywordsSents
        self.tokenizer = tokenizer

    def generateQA(self):
        index = 0
        identifyWords = InVocab(self.vectorModel.wv.vocab,self.ChosenWords)
        FinalWords,FinalNumbers,keywordsSents = identifyWords.VocabChosen(self.keywordsSents)
        FinalNumbers = identifyWords.SelectNumbers(FinalNumbers,keywordsSents)
        self.generateAnswers(keywordsSents,FinalWords)
        AlexaQA = self.finalQA(keywordsSents) # Obtain the final sentences and answers for the Alexa device

        for x in AlexaQA: # Place the answer of questions as a possible option
            AlexaQA[index][2][0].append(AlexaQA[index][0])
            AlexaQA[index][2] = AlexaQA[index][2][0]
            index+=1

        [AlexaQA.append(x) for x in FinalNumbers]
        if(FinalNumbers != "No Numbers"): self.generateNumMCQs(FinalNumbers)

        return AlexaQA

    def generateAnswers(self,keywordsSents,FinalWords): # This function allows the word2vec model to be re-trained
        trained = False

        for x in keywordsSents: # Adding similar words into the array
            sm = []
            text_list = x[1].split()
            phraseTag = nltk.pos_tag(text_list)
            sm = get_similar_words(x[0]) # Use WordNet to find similar words

            if(len(sm) == 0): # WordNet cannot find any similar words -- Use Word2Vec (A more vigorous and lengthy process)
                if(trained == False): # Only train the model once again (this is a large process)
                    trained = True
                    #updatedModel = TrainVectorModel(FinalWords,self.vectorModel,self.tokenizer) # RETRAIN WORD2VEC MODEL (Has not been included in the game due to its lengthy process)
                    #self.vectorModel = updatedModel.updateModel()
                for y in self.vectorModel.wv.vocab:
                    if y == x[0]:
                        simWords = self.vectorModel.most_similar(x[0])
                        x.append(simWords)
            else:
                x.append(sm)

    def finalQA(self,keywordsSents): # A collection of the final sentences and answers
        KeyIndex = 0
        AlexaIndex = 0
        AlexaQA = keywordsSents.copy() # The final questions,answers and options for the Alexa device
        MCQs = generateOptions(self.numMCQoptions)

        for x in keywordsSents:
            if(len(keywordsSents[KeyIndex]) == 3): # If there is a cell/word with MCQ options
                answer = keywordsSents[KeyIndex][0]
                options = keywordsSents[KeyIndex][2]
                AlexaQA[AlexaIndex][2] = MCQs.answerOptions(answer,options)
            else:
                del AlexaQA[AlexaIndex] # There are no MCQ options available for that word
                AlexaIndex -= 1
            AlexaIndex += 1
            KeyIndex += 1
        AlexaQA = self.secureOptions(AlexaQA)

        return AlexaQA

    def secureOptions(self,AlexaQA):
        index = 0
        emptyFlag = -1
        TempQA = AlexaQA.copy()

        for options in AlexaQA:
            if(options[2][0] == emptyFlag or options[2][0] == 'No Options'):
                del TempQA[index]
                index-=1
            index+=1

        return TempQA

    def generateNumMCQs(self, FinalNumbers):

        for x in FinalNumbers:
            isDate,date = self.is_date(x[0])
            if not isDate:
                x.append(self.randomNum(x))
            else:
                x.append(self.alternateDate(date))

    def alternateDate(self, date):
        MaxVal = 0
        MinimumVal = 0
        MinimumRange = MaxRange = 50
        MCQ_Options = []
        yearChange = False
        alteredNum = "date"
        staticNum = "century"

        if(len(date) == 4): # i.e. 2018, 1940 -- Only change the year not the century
            yearChange = True
            staticNum = date[0] + date [1]
            alteredNum = date[2] + date[3]
            MinimumVal = int(alteredNum) - MinimumRange
            MaxVal = int(alteredNum) + MaxRange

        while MinimumVal < 10:
            MinimumRange-= 5
            MinimumVal = int(alteredNum) - MinimumRange

        if(yearChange == True):
            while len(MCQ_Options) < self.numMCQoptions:
                option = randint(MinimumVal,MaxVal)
                if option not in MCQ_Options and option != staticNum: # Option is not the answer
                    MCQ_Options.append(str(staticNum) + str(option))
            MCQ_Options.append(str(staticNum) + str(alteredNum))

        return MCQ_Options

    def is_date(self, string):

        try: # year, month, date
            dateStrObj=parse(string, fuzzy=False)
            isYear, year = self.DateFormats(dateStrObj,string)
            if isYear: return True,year
            else: return False, False
        except:
            return False,False

    def DateFormats(self,dateStrObj, dateString):
        dayBool = False
        dateFormats = []
        year = str(dateStrObj.year)
        month = str(dateStrObj.month)
        day = str(dateStrObj.day)

        firstDate = dateStrObj.strftime("%d %b %Y") # If the date is between the first and 9th ( 01, 02 -- 09)
        if int(firstDate[0] + firstDate[1]) <= 9: dayBool = True

        dateFormats.append(firstDate)
        dateFormats.append(year + ' ' + month)
        dateFormats.append(month + ' ' + year)
        dateFormats.append(year)
        dateFormats.append(day + ' ' + month + ' ' + year)
        dateFormats.append(year + ' ' + month + ' ' + day)
        dateFormats.append(dateStrObj.strftime("%Y %b"))
        dateFormats.append(dateStrObj.strftime("%Y %B"))
        dateFormats.append(dateStrObj.strftime("%b %Y"))
        dateFormats.append(dateStrObj.strftime("%B %Y"))

        dateOne = dateStrObj.strftime("%d %B %Y")
        dateFormats.append(dateOne)
        dateTwo = dateStrObj.strftime("%Y %b %d")
        dateFormats.append(dateTwo)

        if dayBool: # If the date is 01, 02, the parsed string may be 1 september not 01 september
            dateFormats.append(self.alteredDay(dateOne,0).lower())
            dateFormats.append(self.alteredDay(dateTwo,len(dateTwo)-2).lower())
        try:
            if dateString in dateFormats:
                string = dateString.split()
                if year in string:
                    return True,year
            else:
                return False,year
        except:
            return False,False

    def alteredDay(self,string,index):
        newstr = string[:index] + string[index+1:]
        return newstr


    def randomNum(self, number):
        alteredNum = 0
        possibleUses = [] # Possible Numbers to use in the string
        MCQ_Options = []
        AssessedNum = number[0] # The number containing section of the string
        text_list = AssessedNum.split()
        phraseTag = nltk.pos_tag(text_list)

        for phrase,Tag in phraseTag:
            if(Tag == 'CD'):
                pair = []
                pair.append(phrase)
                pair.append(self.StringOrNum(phrase)[1])
                possibleUses.append(pair)

        alteredNum = possibleUses[0][0]
        stringCheck = self.StringOrNum(AssessedNum)[1]

        if stringCheck == 2 and not alteredNum.isdigit(): # Ensures number such as 'eighty thousand' are correctly identified given their separation
            temp = w2n.word_to_num(AssessedNum)
            alteredNum = num2words(temp)

        val = self.StringOrNum(alteredNum)[1]
        if(val == 1):MCQ_Options = self.convertNumbers(True,False,alteredNum,AssessedNum)
        else: MCQ_Options = self.convertNumbers(False,True,alteredNum,AssessedNum)

        return MCQ_Options

    def StringOrNum(self, phrase):
        try: # Check if its a numerical number (i.e. 1,2,3)
            num = num2words(phrase)
            return True,1,num
        except: # Its not a numerical number
            try: # Check its a word number (i.e. one,two,three)
                num = w2n.word_to_num(phrase)
                return True,2,num
            except:
                return False,-1,-1 # -1 represents an error flag

    def convertNumbers(self, isNum,isString,phrase,entirePhrase):
        index = 0
        MCQ_Options = []

        if isString: num = w2n.word_to_num(phrase)
        else: num = int(phrase)

        MinimumVal = num-50
        MaxVal = num+50
        if MinimumVal < 0: MinimumVal=num-10 # The Minimum value cannot go below zero
        if MinimumVal < 0: MinimumVal = 0

        while len(MCQ_Options) < self.numMCQoptions:
            option = randint(MinimumVal,MaxVal)
            if option not in MCQ_Options and option != num:
                MCQ_Options.append(option)
        MCQ_Options.append(num)

        for x in MCQ_Options:
            if isString: MCQ_Options[index] = num2words(x)
            else: MCQ_Options[index] = str(x)
            index += 1

        return MCQ_Options

class Utility(): # Ensures the questions are of the best possible quality
    def __init__(self,AlexaQA):
        self.FinalQAs = AlexaQA

    def isAmbigousMaterial(material): # The string may contain none-sense material such as list items
        count = 0
        for x in material: # Multiple new lines indicate a list element -- This material cannot be used for the Game
            if x == '\n': count+=1
            if x == ':': return True # Suggests subheadings are being used

        if count >= 3: return True
        else: return False

    def cleanData(self):
        self.removeDuplicates()
        self.removeAmibigousSents()
        self.cleanWord2VecMaterial()
        self.removeUnwatedWords()
        self.creatDict()
        return self.FinalQAs

    def removeDuplicates(self):

        index = 1
        for x in self.FinalQAs:
            sentence = x[1]
            for y in self.FinalQAs[index:]:
                tempSent = y[1]
                if sentence == tempSent:
                    self.FinalQAs[index-1] = "Duplicate"
            index+=1

    def removeAmibigousSents(self): # Remove any ambigous sentences that are likely random lists of words
        index = 0
        FinalQACopy = self.FinalQAs.copy()

        for x in self.FinalQAs: # Sentences that do not end with a period are usually words parsed from a list on the website
            sentence = x[1]
            if sentence[len(sentence)-1] != '.':
                del FinalQACopy[index]
                index-=1
            elif(len(x) > 3):
                del FinalQACopy[index][2]
            index+=1

        self.FinalQAs = FinalQACopy

    def cleanWord2VecMaterial(self):
        index = 0
        FinalQACopy = self.FinalQAs.copy()

        for x in self.FinalQAs:
            questionHold = []
            for y in range(len(x[2])):
                if isinstance(x[2][y], tuple):
                    questionHold.append(x[2][y][0]) # Just take the answer, not the vector weighting
                else:
                    questionHold.append(x[2][y])
            FinalQACopy[index][2] = questionHold
            index += 1

        self.FinalQAs = FinalQACopy

    def removeUnwatedWords(self):
        index = 0

        line = ""
        for x in self.FinalQAs:
            identified = False
            subIndex = 0
            keywords = x[0]
            keys = keywords.split()
            sentence = x[1].split()

            for y in keys:
                line = ""
                if y in x[2]:
                    identified = True
                    self.blankWord(index,y,sentence,False)
                subIndex+=1
            if (identified == False):
                try: # Check its a word number (i.e. one,two,three)
                    num = w2n.word_to_num(keywords)
                    word = num2words(num)
                    self.blankWord(index,word,sentence,True)
                except: # This should never happen as the value is checked earlier
                    print("System Error")
                    exit(3)
            index +=1

    def blankWord(self, index, word, sentence, isUnidentified): # Creates a space in the sentences where the answer is
        line = ""
        sentIndex = 0
        self.FinalQAs[index][0] = word
        if isUnidentified:
            word = word.capitalize()
            sentence = self.FinalQAs[index][1]

        for k in range(len(word)):
            line = line + "_"

        if isUnidentified: # Produces blank lines for double phrased answers (i.e. eighty thousand)
            indexes = []
            wordIndex = 0
            for letter in sentence:
                if wordIndex < len(word) and letter == word[wordIndex]:
                    indexes.append(wordIndex)
                elif(len(indexes) == len(word)):
                    sent = []
                    for line in sentence:
                        for char in line:
                            sent.append(char)
                    for index1 in indexes:
                        sent[index1] = "_"
                    sent = ''.join(sent)
                    self.FinalQAs[index][1] = sent
                    return
                elif(wordIndex < len(word)):
                    indexes = []
                wordIndex+=1

        for k in sentence:
            EndLineWord = word + "." # Words at the end of a sentence
            commaWord = word + "," # Words attached with a comma

            if k == word or k == EndLineWord or k == commaWord:
                sentence[sentIndex] = line
                self.FinalQAs[index][1] = ' '.join(sentence)
            sentIndex+=1

    def creatDict(self):
        FinalQAs = []
        for x in self.FinalQAs:
            random.shuffle(x[2]) # Prevents the answer from always being in the last cell
            dict = {
                 "answer": x[0],
                 "sentence": x[1],
                 "options": x[2]
            }
            FinalQAs.append(dict)

        self.FinalQAs = FinalQAs

numOptions = 2 # Number of other multiple choice options -- The user will have 3 options including the answer
AnswerSents = QuestionAnswer(numOptions,vectorModel,ChosenWords,keywordsSents,tokenizer)
AlexaQA = AnswerSents.generateQA()
AlexaCopy = AlexaQA.copy()

index = 0
for x in AlexaCopy:
    if(len(x[1].split()) < 8):
        del AlexaQA[index]
        index-=1
    index+=1

Utils = Utility(AlexaQA)
AlexaQA = Utils.cleanData()

Game.setAlexaQAs(AlexaQA)
Game.playGame()
