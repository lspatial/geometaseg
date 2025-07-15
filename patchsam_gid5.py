import sys
import os 
import shutil 

sys.path.append('.')
from  datagid5.sampling_complexity_testall import parallelSamplingGID


def giddata(comdist=20,tpath=None):
    if tpath is None: 
      tpath = '/geosampling/complexRSPatches/gid5_'+str(comdist)  
    if os.path.exists(tpath):
       shutil.rmtree(tpath)
    os.makedirs(tpath) 
    sampling = parallelSamplingGID(patchsize=256,complexbdist=comdist,ncore=50)
    sampling.samallpatchBFl(tpath)
 
def main():
    comdist=10 
    tpath='/geosampling/complexRSPatches/gid5_'+str(comdist)
    giddata(comdist,tpath)

if __name__=='__main__':
    main()
