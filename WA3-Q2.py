#WA3 - p.313 Q2

from tkinter import filedialog
from tkinter import *
import tkinter
import numpy as np
import math

#my code from Project 1
#open BMP & Obtain RGB Data
#returns rgb data, width and height of image
def readBMP(filename):

    imgData = bytearray()

    with open(filename, 'rb') as image:
        data = image.readlines()
        for i in range(len(data)):
            #print("Row: ", i)
            #print(bytearray(data[i]))
            imgData+=bytearray(data[i])
            

            
    width =(int.from_bytes(imgData[18:22],"little")) #width
    height = (int.from_bytes(imgData[22:26],"little")) #height
    pixelData = imgData[54:(((width*3)+((width*3)%4))*height)+54]

    pixelInt = np.asarray(pixelData)
    #print(len(pixelInt))
    pixelInt = pixelInt.reshape(-1,((width*3)+((width*3)%4)))
    #print(len(pixelInt))
    pixelInt = pixelInt[::1,0:width*3]
    #print(len(pixelInt))
    #print(width, height)
    pixelInt = pixelInt.reshape(-1, 3)
    pixelInt = np.flip(pixelInt)
    #print(len(pixelInt))
    return pixelInt, width, height

#Complete DCT II on each channel
#Converts a data set into 8x8 DCT Coefficients
# based on process described here: (page 2)
#https://www.cse.iitb.ac.in/~ajitvr/CS754_Spring2017/dct_laplacian.pdf
def DCT(data):

    transformData = [[], [], [], [], [], [], [], []]

    for i in range(len(transformData)):
        for j in range(8):

            Ci = 0
            Cj = 0

            if(i == 0):
                Ci = 1/math.sqrt(2)
            else:
                Ci = 1
            if(j == 0):
                Cj = 1/math.sqrt(2)
            else:
                Cj = 1

            Ci = Ci/2
            Cj = Cj/2

            cosineSum = 0
            for p in range(len(data)):
                for q in range(len(data[i])):

                    dataVal = data[p,q]
                    cosine1 = math.cos(((2*p+1)*math.pi*i)/16)
                    cosine2 = math.cos(((2*q+1)*math.pi*j)/16)

                    cosineSum+= dataVal*cosine1*cosine2
                    
            transformData[i].append(round(cosineSum*Ci*Cj,1))
    #print (transformData)
    return transformData

##converts set of DCT cofficients back to shifted pixel data
##based off of 2D DCT-III shown here:
##https://en.wikipedia.org/wiki/JPEG#Decoding
def inverseDCT(data):

    transformData = [ [], [], [], [], [], [], [], []]

    for i in range(len(transformData)):
        for j in range(8):
            
            cosineSum = 0
            for p in range(len(data)):
                for q in range(len(data[p])):
                    Cp = 0
                    Cq = 0

                    if(p == 0):
                        Cp = 1/math.sqrt(2)
                    else:
                        Cp = 1
                    if(q == 0):
                        Cq = 1/math.sqrt(2)
                    else:
                        Cq = 1
                    
                    dataVal = data[p][q]
                   
                    cosine1 = math.cos(((2*i+1)/16)*math.pi*p)
                    cosine2 = math.cos(((2*j+1)/16)*math.pi*q)

                    cosineSum+= dataVal*cosine1*cosine2*Cp*Cq

            transformData[i].append(round(.25*cosineSum,1))
    #print (transformData)

    return transformData

#converts data from Quantized to DTC
# or DTC to Quantized
#lookup table based on Photoshop Save-As JPG 05:
#https://www.impulseadventure.com/photo/jpeg-quantization-lookup.html?src1=10318
def quantizeChrominance(data, inverse=False):
    quantTable = [[13, 13, 17, 27, 20, 20, 17, 17],
                  [13, 14, 17, 14, 14, 12, 12, 12],
                  [17, 17, 14, 14, 12, 12, 12, 12],
                  [27, 14, 14, 12, 12, 12, 12, 12],
                  [20, 14, 12, 12, 12, 12, 12, 12],
                  [20, 12, 12, 12, 12, 12, 12, 12],
                  [17, 12, 12, 12, 12, 12, 12, 12],
                  [17, 12, 12, 12, 12, 12, 12, 12]]
    
    quantResult = [ [], [], [], [], [], [], [], []]
    if(inverse):
        for i in range(len(data)):
            for j in range(len(data[i])):
                quantResult[i].append(int(data[i][j]*quantTable[i][j]))
    else:
        for i in range(len(data)):
            for j in range(len(data[i])):
                resultVal = data[i][j]/quantTable[i][j]
                if(resultVal>=1 or resultVal<=-1):
                    quantResult[i].append(int(resultVal))
                else:
                    quantResult[i].append(0)

    return quantResult

def RGBtoYCbCr(R,G,B):

##    Y = .229*R + .587*G + .114*B
##    Cb = 128 -.1687*R -.3313*G + .5*B 
##    Cr = .5*R + -.4187*G + -.081312*B + 128
    Y = 0.257*R + 0.504*G + 0.098*B + 16
    Cb = -0.148*R - 0.291*G + 0.439*B + 128
    Cr = 0.439*R - 0.368*G - 0.071*B + 128

    return [Y,Cb,Cr]

def YCbCrtoRGB(Y,Cb,Cr):

##    R = Y + 1.402*(Cr-128)
##    G = Y + -.344136*(Cb-128) + -.71436*(Cr-128)
##    B = Y + 1.772*(Cb-128)

    R = 1.164*(Y-16) + 1.596*(Cr-128)
    G = 1.164*(Y-16) - 0.813*(Cr-128) - 0.392*(Cb-128)
    B = 1.164*(Y-16) + 2.017*(Cb-128)

    return [R,G,B]

def subSample(channelData):
    channelData = np.asarray(channelData)
    #subArr = np.array([], dtype="f")
    subArr1 = channelData[::1,::2,::2]
    subArr2 = channelData[::1, 1::2, ::2]

    subArr = (subArr1+subArr2)/2    
##    print(subArr1)
##    print(subArr2)
##    print(subArr)
    
    return subArr

def expandSamples(channelData):

    expandData = np.array([])

    for j in np.nditer(channelData):
        expandData = np.append(expandData,j)
        expandData = np.append(expandData,j)
        
    expandData = expandData.reshape(-1,8)
    expandData = np.repeat(expandData, 2, axis=0)
    expandData = expandData.reshape(-1,8,8)

    #print(expandData)
    return expandData
        
    
#displays BMP on GUI after reading RGB Data
def displayBMP(bmpRGB, imgWidth, imgHeight, path):

    global bmpLabel
    global top

    #(for PBM display, convert to PBM and display)
    print("Converting BMP to PBM for display...")
    pbmPath = savePBMBinary(path, imgWidth, imgHeight, bmpRGB)
    bmpImage = PhotoImage(file=pbmPath)
    bmpLabel = Label(top, image=bmpImage)
    bmpLabel.image=bmpImage
    bmpLabel.pack(side="left")
    top.update()

#displays BMP on GUI after reading RGB Data
def displayIMG(imgRGB, imgWidth, imgHeight, path):
   
    global imgLabel
    global top

    #(for PBM display, convert to PBM and display)
    print("Converting BMP to PBM for display...")
    pbmPath = savePBMBinary(path+"-noY", imgWidth, imgHeight, imgRGB)
    imgImage = PhotoImage(file=pbmPath)
    imgLabel = Label(top, image=imgImage)
    imgLabel.image=imgImage
    imgLabel.pack(side="left")
    top.update()
        
#transform BMP without Y data
def encodeBMP(path, saveto):
    #extract BMP data and display it on GUI
    bmpRGB, width, height = readBMP(path)
    #print(bmpRGB)
    print("BMP Data Read.")
    displayBMP(bmpRGB, width, height, saveto)

    crData = []
    cbData = []
    #convert RGB data to YCbCr color space
    print("Converting to YCbCr Color Space")
    for i in bmpRGB:
        result = RGBtoYCbCr(i[0],i[1],i[2])
        cbData.append(result[1])
        crData.append(result[2])

    cbData = np.asarray(cbData)
    crData = np.asarray(crData)
    cbData = cbData.reshape(-1,8,8)
    crData = crData.reshape(-1,8,8)

    cbSamp = subSample(cbData)
    crSamp = subSample(crData)
    print("Subsampling Data...")
##    for i in crData:
##        resultCr = subSample(i)
##        crSamp = np.append(crSamp, resultCr, axis=0)
##        
##    for i in cbData:
##        resultCb = subSample(i)
##        cbSamp = np.append(cbSamp,resultCb, axis=0)

    cbSamp = cbSamp.reshape(-1,8,8)
    crSamp = crSamp.reshape(-1,8,8)
   
    print("Expanding Data...")
    cbSamp = expandSamples(cbSamp)
    crSamp = expandSamples(crSamp)

    cbSamp = cbSamp.reshape(-1)
    crSamp = crSamp.reshape(-1)

    print("Converting back to RGB...")
    imgRGB = []
    for i in range(len(cbSamp)):
        result = RGBtoYCbCr(128,cbSamp[i],crSamp[i])
        imgRGB.append(result)

    imgRGB = np.asarray(imgRGB)

    print("Displaying Result...")
    displayIMG(imgRGB, width, height, saveto)
    #
    #DCT
    #Quantize
    #Inverse DCT
    #CbCr to RGB
    #display
#My code from Project 1
#saves a file as a PPM version P6
#this is used to display an image on the canvas
def savePBMBinary(fileName, width, height, myData):
    f = open((fileName+".ppm"), "wb")
    f.write("P6 ".encode("ascii"))
    f.write("\n".encode("ascii"))
    f.write(str(width).encode("ascii"))
    f.write(" ".encode("ascii"))
    f.write(str(height).encode("ascii"))
    f.write("\n".encode("ascii"))
    f.write(str(255).encode("ascii"))
    f.write("\n".encode("ascii"))
    for i in range(0, len(myData), width):
        currentRow = myData[i:i+width]
        for j in range(len(currentRow)-1, -1, -1):
            #print(int(currentRow[j,0]))
            f.write(int(currentRow[j,0]).to_bytes(1, "big"))
            f.write(int(currentRow[j,1]).to_bytes(1, "big"))
            f.write(int(currentRow[j,2]).to_bytes(1, "big"))
    print(fileName+".ppm")
    return (fileName+".ppm")

def getFilename(path):
    extension = path[len(path)-4:len(path)]
    path = path[0:len(path)-4]
    slashIndex = path.rfind("/")
    path=path[slashIndex+1:len(path)]
    return path, extension

def fileCallback():
    global pbmDisplay
    global bmpLabel
    global imgLabel
    #clear canvases
    bmpLabel.image=None
    imgLabel.image=None
    bmpLabel.pack_forget()
    imgLabel.pack_forget()
    bmpLabel = Label(top)
    imgLabel = Label(top)

    #get file path in BMP or IMG
    filename = filedialog.askopenfilename(initialdir="/", title="Select BMP or IMG")
    saveto, extType = getFilename(filename)

    print("Encoding BMP to IMG...")
    encodeBMP(filename, saveto)



    return

pbmDisplay = True
global bmpLabel
global imgLabel
global top
if __name__ == '__main__':
    global bmpLabel
    global imgLabel
    global top
    top = tkinter.Tk()
    top.title("IMG File Reader & Writer")
    
    bmpLabel = Label(top)
    imgLabel = Label(top)
    
    fileButton = tkinter.Button(top, text="Open File", command=fileCallback)
    fileButton.pack()
    pbmDisplay = None
    
    top.mainloop()
