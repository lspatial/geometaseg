
import os 
import re 
from esda.moran import Moran
import pysal as ps 
from libpysal.weights import lat2W 
import numpy as np 
import sys 
import pandas as pd 

def getduty(tcls):
  data_path='/geosampling/complexRSPatches/gid5_10_mcomplex/'+str(tcls)
  listall = [f for f in os.listdir(data_path) if re.search('\\.npz$', f)]
  print(len(listall))
  return listall 


if __name__=='__main__':
  _,cls,ifrom,ito=sys.argv
  ifrom=int(ifrom)-1
  ito=int(ito) 
  icls=int(cls)
  listall=getduty(icls)
  for i in range(ifrom,ito):
      tfl=listall[i]
      print(i,tfl)
      tfl='/geosampling/complexRSPatches/gid5_10_mcomplex/'+str(icls)+'/'+tfl
      datan=np.load(tfl,allow_pickle=True) 
      covs=['m21', 'e21','p21','m15','e15','p15', 'm11','e11', 'p11', 'm7','e7', 'p7', 'm5', 'e5','p5']
      cmd="pd.DataFrame({'icls':icls,'tfl':tfl"
      for c in covs:
        cmd = cmd + ",'"+c+"_mean':np.nanmean(datan['"+c+"'])"
      cmd = cmd + "},index=["+str(i)+"])"
      aresult=eval(cmd)
      if i==ifrom:
          allresult=aresult
      else:
          allresult = pd.concat([allresult, aresult])
  outarget='/geosampling/complexRSPatches/gid5_10_mcomplex_sum/'+str(icls)
  if not os.path.exists(outarget):
     os.makedirs(outarget)
  tfl=outarget+'/f_'+str(ifrom)+'_'+str(ito)+'.csv'
  allresult.to_csv(tfl,index=False) 
  
