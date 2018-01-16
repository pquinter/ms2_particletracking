from scipy import *
from scipy import ndimage
from scipy.misc import *

def GaussianMaskFit2(im,coo,s,optLoc=1,bgSub=2,winSize=13,convDelta=.01,nbIter=20):
  """Applies the algorithm from [Thompson et al. (2002) PNAS, 82:2775].
Parameters:
- im: a numpy array with the image
- coo: approximate coordinates (in pixels) of the spot to localize and measure
- s: width of the PSF in pixels
- optLoc: If 1, applied the iterative localization refinement algorithm, starting with the coordinates provided in coo. If 0, only measures the spot intensity at the coordinates provided in coo.
- bgSub: 0 -> no background subtraction. 1 -> constant background subtraction. 2 -> tilted plane background subtraction.
- winSize: Size of the window (in pixels) around the position in coo, used for the iterative localization and for the background subtraction.
- convDelta: cutoff to determine convergence, i.e. the distance (in pixels) between two iterations
- nbIter: the maximal number of iterations.

Returns
- the intensity value of the spot.
- the corrdinates of the spot.

If convergence is not found after nbIter iterations, return 0 for both intensity value and coordinates.
"""
  from scipy import optimize; coo=array(coo);
  for i in range(nbIter):
    if not prod(coo-winSize/2.>=0)*prod(coo+winSize/2.<=im.shape[::-1]): return 0.,r_[0.,0.], 0.
    winOrig=(coo-int(winSize)/2).astype(int)
    i,j=meshgrid(winOrig[0]+r_[:winSize],winOrig[1]+r_[:winSize]);
    N=exp(-(i-coo[0])**2/(2*s**2)-(j-coo[1])**2/(2*s**2))/(2*pi*s**2)
    # get 13x13 window around spot center, raw image.
    # equivalent to: im[winOrig[1]:winOrig[1]+winSize:,winOrig[0]:winOrig[0]+winSize]*1.
    S=im[:,winOrig[0]:winOrig[0]+winSize][winOrig[1]:winOrig[1]+winSize]*1.
    if bgSub==2:
      xy=r_[:2*winSize]%winSize-(winSize-1)/2.
      bgx=polyfit(xy,r_[S[0],S[-1]],1); S=(S-xy[:winSize]*bgx[0]).T;
      bgy=polyfit(xy,r_[S[0],S[-1]],1); S=(S-xy[:winSize]*bgy[0]).T;
      bg=mean([S[0],S[-1],S[:,0],S[:,-1],]); S-=bg
      bg=r_[bg,bgx[0],bgy[0]]
    if bgSub==1:
      bg=mean([S[0],S[-1],S[:,0],S[:,-1],]); S-=bg
    S=S.clip(0) # Prevent negative values !!!!
    if optLoc:
      SN=S*N; ncoo=r_[sum(i*SN),sum(j*SN)]/sum(SN)
      #ncoo=ncoo+ncoo-coo # Extrapolation
      if abs(coo-ncoo).max()<convDelta: return sum(SN)/sum(N**2),coo,bg
      else: coo=ncoo
    else: return sum(S*N)/sum(N**2),coo,bg
  return 0.,r_[0.,0.], 0.


sHS=fftpack.fftshift # Swap half-spaces. sHS(matrix[, axes]). axes=all by default
def hS(m,axes=None):
    if axes==None: axes=range(rank(m))
    elif type(axes)==int: axes=[axes]
    elif axes==[]: return m
    return hS(m.swapaxes(0,axes[-1])[:m.shape[axes[-1]]/2].swapaxes(0,axes[-1]),axes[:-1])

def sHSM(m,axes=None):
    if axes==None: axes=range(rank(m))
    elif type(axes)==int: axes=[axes]
    m=m.swapaxes(0,axes[0]); max=m[1]+m[-1]; m=(m+max/2)%max-max/2; m=m.swapaxes(0,axes[0])
    return sHS(m,axes)


def bpass(im,r1=1.,r2=1.7):
  ker1x=exp(-(sHS(sHSM(r_[:im.shape[1]]))/r1)**2/2); ker1x/=sum(ker1x); fker1x=fft(ker1x);
  ker1y=exp(-(sHS(sHSM(r_[:im.shape[0]]))/r1)**2/2); ker1y/=sum(ker1y); fker1y=fft(ker1y);
  ker2x=exp(-(sHS(sHSM(r_[:im.shape[1]]))/r2)**2/2); ker2x/=sum(ker2x); fker2x=fft(ker2x);
  ker2y=exp(-(sHS(sHSM(r_[:im.shape[0]]))/r2)**2/2); ker2y/=sum(ker2y); fker2y=fft(ker2y);
  fim=fftpack.fftn(im)
  return fftpack.ifftn((fim*fker1x).T*fker1y-(fim*fker2x).T*fker2y).real.T



