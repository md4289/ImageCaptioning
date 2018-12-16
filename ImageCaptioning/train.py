#Computer Vision - Course Project
#Instructor - Prof. Rob Fergus
#Image Captioning using ResNet as Encoder and LSTM as Decoder
#Dataset: MSCOCO (Microsoft Common Object in Context)
#Author: Mohith Damarapati | md4289 | N10540205 | New York University


#Torch and Torchvision
import torch 
from torchvision import transforms
#To save and load vocabSet
import pickle
#Data Loader
from load_data import load_data
from PIL import Image
import torch.nn
from torch.nn.utils.rnn import pack_padded_sequence
import numpy
import matplotlib.pyplot as plt
import model

#To run on GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




# PATHS: ** Change Paths while running on hpc cluster **



def cap_reverse(caption, length):
	caption=caption.numpy()
	caps = []
	for i in range(length):
		caps.append(vocabularySet2[caption[i]])
	return caps

with open('vocabSet.pkl', 'rb') as f:
			vocabularySet = pickle.load(f)

print("Loaded Vocabulary Set")

with open('vocabSet2.pkl', 'rb') as f:
			vocabularySet2 = pickle.load(f)


print("Loaded Reverse Vocabulary Set")
modelPath = "models/"
imagesPath = "../data/images/"
captionsPath = "../data/annotations/captions_train.json"

#Hyper Parameters  -  TUNABLE
lstmLayers = 3
lstmHiddenStates = 512
wordEmbeddings = 256
epochs = 5
batchSize = 64
learningRate = 0.001

cnn = model.EncoderCNN(wordEmbeddings).to(device)
lstm = model.DecoderRNN(wordEmbeddings, lstmHiddenStates, len(vocabularySet), lstmLayers).to(device)

criterion = torch.nn.CrossEntropyLoss()
parameters = list(lstm.parameters()) + list(cnn.linear.parameters()) + list(cnn.bn.parameters())
optimizer = torch.optim.Adam(parameters, lr=learningRate)

#Preprocessing of Image data
transform = transforms.Compose([ transforms.RandomCrop(224), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


data_loader = load_data(vocabularySet, imagesPath, captionsPath, batchSize, transform, shuffle=True, num_workers=2)

lossplot = []

for e in range(epochs):
	for i, (imageBatch, captionBatch, lengthBatch) in enumerate(data_loader):
		imageBatch = imageBatch.to(device)
		captionBatch = captionBatch.to(device)

		'''
		Sample Captions and Images Visualisation - Pasted In Report

		print(image[i].numpy().transpose(2,1,0).shape)
		plt.imshow(image[i].numpy().transpose(2,1,0))
		plt.show()
		origCaption = cap_reverse(caption[i], length[i])
		print(origCaption)
		'''

		packedCaptions = pack_padded_sequence(captionBatch, lengthBatch, batch_first=True)[0]
		#print(packedCaptions)

		cnnFeatures = cnn(imageBatch)
		lstmOutput = lstm(cnnFeatures, captionBatch, lengthBatch)
		loss = criterion(lstmOutput, packedCaptions)

		lstm.zero_grad()
		cnn.zero_grad()
		loss.backward()
		optimizer.step()

		if i%100 == 0:
			print("Epoch: "+ str(e) + " Batch: "+ str(i) + " Loss: "+ str(loss.item()))


	torch.save(lstm.state_dict(),modelPath + 'decoder' + str(e) + '.ckpt')
	torch.save(cnn.state_dict(),modelPath + 'encoder' + str(e) + '.ckpt')

	