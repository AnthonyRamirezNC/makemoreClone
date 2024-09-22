import torch
import matplotlib.pyplot as plt
class makeMore:
    def __init__(self):
        #open the dataset with each line split
        self.words = open("names.txt", 'r').read().splitlines()
        self.generateCharacterMappings()
        self.createBigramTensor()

    def generateCharacterMappings(self):
        #get all character values possible
        self.chars = sorted(list(set(''.join(self.words))))
        #create String to integer lookup table
        self.StrToI = {s:i+1 for i,s in enumerate(self.chars)}
        #insert special characters
        self.StrToI['.'] = 0
        #create Integer to String lookup table
        self.IToStr = {i:s for s, i in self.StrToI.items()}


    def generateBigram(self):
        #iterate through each word
        for w in self.words:
            #get all the bigrams in current word
            chs = ['.'] + list(w) + ['.']
            for ch1, ch2 in zip(chs, chs[1:]):
                ix1 = self.StrToI[ch1]
                ix2 = self.StrToI[ch2]
                self.N[ix1, ix2] += 1

    def createBigramTensor(self):
        self.N = torch.zeros((27,27), dtype = torch.int32)
        self.generateBigram()
        self.createProbabilityTensor()

    def createProbabilityTensor(self):
        self.P = self.N.float()
        for i in range(27):
            self.P[i] = self.N[i] / self.N[i].sum()
            

    def plotBigramTensor(self):
        plt.figure(figsize=(32,32))
        plt.imshow(self.N, cmap='Blues')
        for i in range(27):
            for j in range(27):
                chrStr = self.IToStr[i] + self.IToStr[j]
                plt.text(j, i, chrStr, ha = "center", va = "bottom", color = 'gray')
                plt.text(j, i, self.N[i, j].item(), ha="center", va="top", color="gray")
        plt.axis('off')
        plt.show()

    def plotProbabilityTensor(self):
        plt.figure(figsize=(64,64))
        plt.imshow(self.P, cmap='Blues')
        for i in range(27):
            for j in range(27):
                chrStr = self.IToStr[i] + self.IToStr[j]
                plt.text(j, i, chrStr, ha = "center", va = "bottom", color = 'gray')
                plt.text(j, i, round(self.P[i, j].item(), 3), ha="center", va="top", color="gray")
        plt.axis('off')
        plt.show()

    def sampleModel(self):
        g = torch.Generator().manual_seed(21747483647)
        ix = 0
        output = []
        while True:
            p = self.P[ix]
            #using multinomial function to assign next char given probabbility distribution
            ix = torch.multinomial(p, 1, replacement = True, generator=g).item()
            output.append(self.IToStr[ix])
            #Selected end token so break while loop
            if ix == 0:
                break

        return ''.join(output)

            

            
        


instance = makeMore()
instance.generateBigram()
instance.createBigramTensor()
#instance.plotBigramTensor()
#print(instance.sampleModel())
instance.plotProbabilityTensor()