#Computer Vision - Course Project
#Instructor - Prof. Rob Fergus
#Image Captioning using ResNet as Encoder and LSTM as Decoder
#Dataset: MSCOCO (Microsoft Common Object in Context)
#Author: Mohith Damarapati | md4289 | N10540205 | New York University

#Pycocotools is a part of COCO API built by Microsoft to handle COCO dataset.
from pycocotools.coco import COCO
#Natural Language ToolKit
import nltk
#Torch and Torchvision
import torch
import torchvision.transforms as transforms
#Pytorch Dataloader
import torch.utils.data as data
#Image Library
from PIL import Image 



class MSCOCODataset(data.Dataset):

	def __init__(self, vocabularySet, imagePath, captionPath, transform):

		self.vocabularySet = vocabularySet
		self.imagePath = imagePath
		self.captionPath = captionPath
		self.transform = transform
		self.capdata = COCO(captionPath)
		#Creating list of whole dataset - all captions
		self.captions = list(self.capdata.anns.keys())

	def __getitem__(self, i):

		capdata = self.capdata
		captions_ind = self.captions[i]
		vocabularySet = self.vocabularySet

		imageID = capdata.anns[captions_ind]['image_id']
		#captionID = capdata.anns[captions_ind]['caption_id']
		caption = capdata.anns[captions_ind]['caption']

		#Get the image path from image object
		#loadImgs() is a method in pycocotools to get image object
		imageObj = capdata.loadImgs(imageID)
		imgPath = imageObj[0]['file_name']

		#Open image and perform preprocessing
		img = Image.open(self.imagePath+imgPath).convert('RGB')
		img = self.transform(img)
		#img = torch.Tensor(img) 

		final_caption = []
		#Tokenizing the caption
		caption_wordlist = nltk.tokenize.word_tokenize(str(caption).lower())


		final_caption.append(vocabularySet['<s>'])
		for cap in caption_wordlist:
			if cap in vocabularySet:
				final_caption.append(vocabularySet[cap])
			else:
				final_caption.append(vocabularySet['<u>'])

		final_caption.append(vocabularySet['<e>'])
		final_caption = torch.Tensor(final_caption)
		return img, final_caption

	def __len__(self):
		return len(self.captions)

def my_collate(data):
	data.sort(key=lambda x: len(x[1]), reverse=True)
	images, captions = zip(*data)
	images = torch.stack(images, 0)
	lengths = [len(cap) for cap in captions]
	targets = torch.zeros(len(captions), max(lengths)).long()
	for i, cap in enumerate(captions):
		end = lengths[i]
		targets[i, :end] = cap[:end]        

	return images, targets, lengths

def load_data(vocabularySet, imagePath, captionPath, batch_size, transform, shuffle, num_workers):

	dataset = MSCOCODataset(vocabularySet = vocabularySet, imagePath=imagePath, captionPath=captionPath, transform=transform)
	#Inbuild Pytorch function to load data with given batch size
	dataLoader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=my_collate)
	return dataLoader
