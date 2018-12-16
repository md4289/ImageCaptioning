#Computer Vision - Course Project
#Instructor - Prof. Rob Fergus
#Image Captioning using ResNet as Encoder and LSTM as Decoder
#Dataset: MSCOCO (Microsoft Common Object in Context)
#Author: Mohith Damarapati | md4289 | N10540205 | New York University


#Pycocotools is a part of COCO API built by Microsoft to handle COCO dataset.
from pycocotools.coco import COCO
#Counter - Keep track of word frequencies
from collections import Counter
#Natural Language ToolKit
import nltk
#To save and load VocabSet
import pickle

# PATHS: ** Change Paths while running on hpc cluster **

captionPath = "../data/annotations/captions_train.json"
vocabPath = "../data/vocabulary.pkl" 

#To generate quality captions, avoid certain rarely occuring words 
#I am considering the minimum required word frequence to be 5  -  TUNABLE
#Counter keeps track of overall frequency of words accross all the captions 
minwordFreq = 4
count = Counter()


data = COCO(captionPath)

#Exploiting Pycocotools to get insights about data
print("Total Annotations: " + str(len(data.anns.keys())))
print("Total Images: " + str(len(data.imgs.keys())))

#Will Print:
#Total Annotations: 414113 - 5 Captions on an average per image {Can be more or less}
#Total Images: 82783

#COCO API - anns is a dictionary with mappings from key to {imageID, captionID, caption}
  
#To print few of those captions
print(data.anns.items()[0][1]['caption'])
print(data.anns.items()[5][1]['caption'])
print(data.anns.items()[10][1]['caption'])

captions_tokenized_list = []
for (i, key) in enumerate(data.anns.keys()):
	caption = str(data.anns[key]['caption'])
	caption = caption.lower()
	#Convert string to list of words
	captions_wordlist = nltk.tokenize.word_tokenize(caption.lower())
	count.update(captions_wordlist)
	#print(i)
	captions_tokenized_list.append(captions_wordlist)

#Printing Captions List and it's length
print(captions_tokenized_list)
print(len(captions_tokenized_list))

#Find Maximum Length Caption
caplen = []
for cap in captions_tokenized_list:
	caplen.append(len(cap))

maxcaplen = max(caplen)

#For checking if padding works
'''for j in captions_tokenized_list:
	for i in range(max(caplen)-len(j)):
			j.append('<pad>')
	print(len(j))'''


#Prints 57 - max caplen

#Sorting words based on their occurrence frequency
[(l,k) for k,l in sorted([(j,i) for i,j in count.items()])]

'''
TOP - 10 word occurrences (Just an insight)

'a': 684577, 
'.': 310919, 
'on': 150675, 
'of': 142760, 
'the': 137981, 
'in': 128909, 
'with': 107703, 
'and': 98754, 
'is': 68686, 
'man': 51530
'''

#Append only words whose frequency is more than minimum word frequency which is 5
words = []

#Additional words - Padding, start, end, and unknown
words.append('<pad>')
words.append('<s>')
words.append('<e>')
words.append('<u>')



for (word, c) in count.items():
	if c >= minwordFreq:
		words.append(word)




print("Words in Vocabulary:	" + str(len(words)))

#Converting Vocabset to dictionary - Represent words as indeces
vocabularySet = {k: v for v, k in enumerate(words)}
vocabularySet2 = {v: k for v, k in enumerate(words)}

#Printing
print(vocabularySet['<u>'])
print(vocabularySet['<e>'])
print(vocabularySet['is'])

#Save VocabSet to a pickle file

with open('vocab/vocabSet.pkl', 'wb') as f:
	pickle.dump(vocabularySet,f)

print("Vocabulary Set Pickled!")

with open('vocab/vocabSet2.pkl', 'wb') as f:
	pickle.dump(vocabularySet2,f)

print("Reverse Vocabulary Set Pickled!")

print(vocabularySet['<pad>'])
print(vocabularySet['<s>'])