#Computer Vision - Course Project
#Instructor - Prof. Rob Fergus
#Image Captioning using ResNet as Encoder and LSTM as Decoder
#Dataset: MSCOCO (Microsoft Common Object in Context)
#Author: Mohith Damarapati | md4289 | N10540205 | New York University

#To retrieve back the vocabulary set
import pickle
#Data Loader
from load_data import load_data
#Torch and Torchvision
import torch
from torchvision import transforms
#Plot Losses
import matplotlib.pyplot as plt 
#Model File
import model 
#NLP Tool Kit
import nltk
#To compute BLEU Score
from nltk.translate.bleu_score import sentence_bleu
#Pycoco tools API
from pycocotools.coco import COCO
#Image Library
from PIL import Image
#To run on GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torch.nn as nn

lstmLayers = 4
lstmHiddenStates = 512
wordEmbeddings = 256
epochs = 5
batchSize = 64
learningRate = 0.001

def cap_reverse(caption, length):
	caption=caption.numpy()
	caps = []
	for i in range(length):
		caps.append(vocabularySet2[caption[i]])
		if(vocabularySet2[caption[i]] == '<e>'):
			return caps
	return caps


with open('vocabSet.pkl', 'rb') as f:
			vocabularySet = pickle.load(f)

print("Loaded Vocabulary Set")

with open('vocabSet2.pkl', 'rb') as f:
			vocabularySet2 = pickle.load(f)

print("Loaded Reverse Vocabulary Set")

modelsPath = "LSTM4Models/"
imagesPath = "../data/val2014/"
captionsPath = "../data/annotations/captions_val.json"

cnnEn = model.EncoderCNN(wordEmbeddings).eval()  
lstmDe = model.DecoderRNN(wordEmbeddings, lstmHiddenStates, len(vocabularySet), lstmLayers)
cnnEn = cnnEn.to(device)
lstmDe = lstmDe.to(device)

valData = COCO(captionsPath)

#Exploiting Pycocotools to get insights about data
print("Total Annotations: " + str(len(valData.anns.keys())))
print("Total Images: " + str(len(valData.imgs.keys())))

#Visualise 
print(valData.imgToAnns[393212])

for (i, key) in enumerate(valData.imgToAnns.keys()):
	origCaptionSet = []
	for rec in valData.imgToAnns[key]:
		origCaptionSet.append(rec['caption'])

	break

#Print Lenght of Val Dataset
print(len(valData.imgs))

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), 
							 (0.229, 0.224, 0.225))])


BLEU = []
for x in range(4):

	cnnEn.load_state_dict(torch.load(modelsPath + "encoder_"+ str(x) + ".ckpt"))
	lstmDe.load_state_dict(torch.load(modelsPath+ "decoder_"+str(x)+".ckpt"))

	bleuCum1 = 0.
	bleuCum2 = 0.
	bleuCum3 = 0.
	bleuCum4 = 0.
	count = 0

	for (i, key) in enumerate(valData.imgs.keys()):
		path = (valData.imgs[key]['file_name'])
		#print(path)
		image = Image.open(imagesPath + str(path)).convert('RGB')
		image = image.resize([224, 224], Image.LANCZOS)
		
		if transform is not None:
			image = transform(image).unsqueeze(0)
		
		origCaptionSet = []
		for rec in valData.imgToAnns[key]:
			#Convert string to list of words
			#print(rec)
			captions_wordlist = nltk.tokenize.word_tokenize(str(rec['caption']).lower())
			captions_wordlist = ['<s>'] + captions_wordlist 
			captions_wordlist.append('<e>')
			#print(captions_wordlist)
			origCaptionSet.append(captions_wordlist)

		image = image.to(device)

		cnnFeatures = cnnEn(image)
		genCaption = lstmDe.sample(cnnFeatures)
		genCaption = genCaption[0].cpu()

		genCaption = cap_reverse(genCaption, len(genCaption))

		#print(genCaption)
		#print(origCaptionSet) 

		bleu1Score = sentence_bleu(origCaptionSet, genCaption, weights=(1,0,0,0))
		bleu2Score = sentence_bleu(origCaptionSet, genCaption, weights=(0.5,0.5,0,0))
		bleu3Score = sentence_bleu(origCaptionSet , genCaption, weights=(0.33,0.33,0.33,0)) 
		bleu4Score = sentence_bleu(origCaptionSet , genCaption) 


		bleuCum4 = bleuCum4 + bleu4Score
		bleuCum3 = bleuCum3 + bleu3Score
		bleuCum2 = bleuCum2 + bleu2Score
		bleuCum1 = bleuCum1 + bleu1Score
		
		count = count + 1
		print("Epoch:" + str(x))
		print("Current Blue4:")
		print(bleuCum4/count)
		


	print("Bleu4:")
	B4 = bleuCum4/len(valData.imgs)
		
	print("Bleu3:")
	
	B3 = bleuCum3/len(valData.imgs)

	print("Bleu2:")

	B2 = bleuCum2/len(valData.imgs)

	print("Bleu1:")
	
	B1 = bleuCum1/len(valData.imgs)

	BLEU.append((B1, B2, B3, B4))	

print(BLEU)
		


	

