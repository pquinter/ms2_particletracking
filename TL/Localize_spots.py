#########################################################################################################
#########################################################################################################
import os
import sys
from misc import *
from spotDetection_functions import *
from numpy import *
from PIL import ImageFilter
from PIL import Image
from scipy import optimize
from tifffile import *
#########################################################################################################
#########################################################################################################

pathIn = "..."
outputfolder = "..."


#########################################################################################################
#### make file list and output directory
#########################################################################################################

if pathIn[-1] != "/": pathIn += "/"
if outputfolder[-1] != "/": outputfolder += "/"


date = pathIn.split("/")[-3]
expName = pathIn.split("/")[-2]

pathOut = outputfolder+date+"_"+expName+"/"

if not os.path.exists(pathOut):
    os.makedirs(pathOut)

lfn = lfn = os.listdir(pathIn)
if ".DS_Store" in lfn: lfn.remove(".DS_Store")
if "display_and_comments.txt" in lfn: lfn.remove("display_and_comments.txt")
lfn.sort()
if processAll == 0:
    lfn = lfn[0:nrFilesToAnalyze]



##### PARAMETERS ######

#### spot threshold:
threshold_cy3=75 # Threshold in intensity

thresholdSD=0; # Threshold in number of standard deviations

psfPx=1.05; # PSF width in pixels
maxDist=3.; # Maximal distance tolerated between guess and fit (in PSF width)
minSeparation=.1; # Minimal distance tolerated between two spots (in PSF width)


par = ["Threshold_cy3="+str(threshold_cy3),"psfPx="+str(psfPx),"MaxDist="+str(maxDist),"minSeparation="+str(minSeparation)]

#######################


for fn in range(0, len(lfn)):
    if not os.path.exists(pathOut+date+"_"+lfn[fn]+"_loc_results_cy3.txt"):
        # Input file
        imagefile = pathOut+date+"_"+lfn[fn]+"_max.tif"

        # Output files
        fnTxt=pathOut+date+"_"+lfn[fn]+"_loc_results_cy3.txt"
        fnTif=pathOut+date+"_"+lfn[fn]+"_loc_results_cy3.tif"
        
        #im=tiff2array(imagefile); im=im[0]*1.                # Raw image
        imtmp = pil.open(imagefile)
        size = imtmp.size
        imtmp.seek(0)                        # 0 for 1st channel, 1 for second channel, 2 for 3rd channel
        im = array(imtmp.convert("I").getdata()).reshape(size)
        im=im*1.
        imBpass=bpass(im,.8,psfPx)                           # Band-passed image
        #imBinary=(imBpass>thresholdSD*var(imBpass)**.5)*1.   # Binary image
        imBinary=(imBpass>threshold_cy3)*1.
        
        # Find all the connex objects
        objects=ndimage.find_objects(ndimage.label(imBinary)[0])
        # Determine their centers as initial guesses for the spot locations
        cooGuess=array([[r_[obj[1]].mean(),r_[obj[0]].mean()] for obj in objects])

        fitResults=[]
        for i in range(cooGuess.shape[0]):
            intensity,coo,tilt=GaussianMaskFit2(im,cooGuess[i],psfPx, winSize = 5)
            # Keep only if it converged close to the inital guess
            if intensity!=0 and sum((coo-cooGuess[i])**2)<(maxDist*psfPx)**2:
                # Remove duplicates
                if sum([sum((coo-a[1:3])**2)<(minSeparation*psfPx)**2 for a in fitResults])==0:
                    fitResults.append(r_[intensity,coo,tilt])

        # Save the results in a text file
        # columns are: integrated intensity, x position, y position, offset of tilted plane, x tilt, y tilt
        savetxt(fnTxt,fitResults, delimiter = "\t")

        # Compute and save image results as a tif file
        imSpots=im*0.
        for a in fitResults: imSpots[int(a[2]+.5),int(a[1]+.5)]=1
        
        im2save = asarray([((a-a.min())/(a.max()-a.min())) for a in [imBpass,imBinary,imSpots]]).astype('uint16')
        im2save2 = concatenate((int16(im).reshape(1,im.shape[0],im.shape[1]),int16(im2save)),axis = 0)
        imsave(fnTif, im2save2)

        print "\n\n*** Fount %d spots. ***\nResults save in '%s' and '%s'."%(len(fitResults),fnTxt,fnTif)
        #par.append("Image "+str(fn)+" has threshold: "+str(thresholdSD*var(imBpass)**.5))
        par.append("Image "+str(fn)+" has threshold: "+str(threshold_cy3))


