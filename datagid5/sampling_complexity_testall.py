import numpy as np
import cv2
import os
import gc
import math
import shutil
import pandas as pd
import sys
import rasterio 
import re 
from rasterio.enums import Resampling

sys.path.append('..')

import math
from multiprocessing import Process
import multiprocessing
from numpy import random  


class parallelSamplingGID:

    def __init__(self,patchsize=512,complexbdist=15,ncore=5):
        self.imgpath = '/rsdatasys/GID/image_RGB'
        self.labpath = '/rsdatasys/GID/label_5classes' 
        self.ncore=ncore
        self.patchsize=patchsize 
        self.complexbdist=complexbdist 
        included_extensions = ['tif']
        file_names = [fn for fn in os.listdir(self.imgpath)
              if any(fn.endswith(ext) for ext in included_extensions)]
        self.imgids=file_names
        clsmap={'r':[0,0,255,0,0,255],'g':[0,0,0,255,255,255],'b':[0,255,0,0,255,0],
           'name':['background','water','buildup','farmland','forest','meadow'],
           'key':[0,1,2,3,4,5]}
        self.clsmapDf=pd.DataFrame(clsmap) 
        self.clsmaposample={0:255,1:255,2:255,3:255,4:255,5:255} 
        print('self.imgids:',len(self.imgids))
#        random.seed(100)
   
    def isPositiveInstance(self,multout,amask,tsize,ifm,it,jfm,jt,border_size=0):
        tsize_r,tsize_c=tsize
        if it > tsize_r:
            it = tsize_r
            ifm = tsize_r -(it-ifm)
        if jt > tsize_c:
            jt = tsize_c
            jfm = tsize_c -(jt-jfm)
        if multout:
            ysam = amask[ifm:(it + border_size * 2), jfm:(jt + border_size * 2), :]
        else:
            ysam = amask[ifm:(it + border_size * 2), jfm:(jt + border_size * 2)]
        ispositive = np.sum(ysam)  > 0.000001
        return ispositive

    def val2MissVal(self,inarr,nsam,nodata=256,threshld=0.1):
        inarrSum=np.nanmean(inarr,axis=2)  
        if str(nodata)=='nan':
          nodata=256
        pcnt=np.count_nonzero(inarrSum==nodata)
        prop=(1.0*pcnt)/(1.0*nsam)
        if prop>threshld:
          return True 
        pcnt=np.count_nonzero(np.isnan(inarr))
        prop=(1.0*pcnt)/(1.0*nsam*3)
        if prop>0.0001:
          print('prop: no data >0.0001+++++++++++++++++++++++++++++++++++++++=')
          return True 
        if np.ma.count_masked(inarr)>0:
          print('masked!!!!!!!')
          return True
        return False 

    def overSampling(self,img_id,threshold,nodata,nsample,undersamplingstep,
                     tpath,imgref,tmaskref,tsize,border_size,patchsize,rstep,cstep):
        lastPositive=True
        counter=1
        tsize_r,tsize_c=tsize
        tnsample=(patchsize+border_size * 2)*(patchsize+border_size * 2)
        #sprint("tnsample:",tnsample,",side:",patchsize+border_size * 2)
        imgid_notif=re.sub('.tif','',img_id) 
        tpath_img=tpath + "/" + imgid_notif
        if not os.path.exists(tpath_img):
                os.mkdir(tpath_img)
        for i in range(0, tsize_r + 1, rstep):
            for j in range(0, tsize_c + 1, cstep):
                ifm = i
                jfm = j
                it = i + patchsize + border_size * 2
                jt = j + patchsize + border_size * 2
                if it > tsize_r:
                    it = tsize_r
                    ifm = tsize_r - patchsize-border_size * 2
                if jt > tsize_c:
                    jt = tsize_c
                    jfm = tsize_c - patchsize-border_size * 2
                ysam = tmaskref[ifm:it, jfm:jt]
                x = imgref[ifm:it, jfm:jt, :]
                y = tmaskref[ifm:it, jfm:jt]
                ispositive = (np.sum(ysam) / nsample) >= threshold 
               # print("x shape:",x.shape[0],",",x.shape[1])
                if not self.val2MissVal(x,tnsample,nodata):  
                    if ispositive: # or not lastPositive:
                        mfile = tpath_img + "/" + imgid_notif + "-" + str(counter) + ".npz"
                        np.savez(mfile, img=x, mask=y)
                        counter += 1 
                del x, y
                gc.collect()
                ispos_undersampling = self.isPositiveInstance(False, tmaskref, tsize,i, i + undersamplingstep, j, j + undersamplingstep,
                                            border_size=0)
                if ispositive or lastPositive or ispos_undersampling:
                    if not ispositive:
                        lastPositive=False
                    else:
                        if not lastPositive:
                            lastPositive=True
                else:
                    j=j+undersamplingstep-cstep

    def sampatchBFl(self, img_id,key, sampleScheme, tpath):
        scaling = sampleScheme['scaling']
        patchsize = sampleScheme['patchsize']
        border_size = sampleScheme['border_size']
        oversamplingstep = sampleScheme['oversamplingstep']
        undersamplingstep = sampleScheme['undersamplingstep']
        
        imgfl = self.imgpath + '/' + img_id 
        labelfl =self.labpath + '/' + re.sub('.tif','_label.tif',img_id)
        if not os.path.isfile(labelfl):
          print('labelfl:',labelfl,'\n not exits!!!')
          return 
        with rasterio.open(imgfl, 'r+') as ds:
          if scaling==1:
            img = ds.read()
          else: 
            img = ds.read(
                out_shape=(
                    ds.count,
                    int(ds.height * scaling),
                    int(ds.width * scaling)
                ),
                resampling=Resampling.bilinear
            )
        with rasterio.open(labelfl, 'r+') as dsmask:
          if scaling==1:
            tmask = dsmask.read(masked=True) 
          else: 
            tmask = dsmask.read(
                   out_shape=(
                    dsmask.count,
                    int(dsmask.height * scaling),
                    int(dsmask.width * scaling)
                ),
                resampling=Resampling.bilinear
          )
           # read all raster values
        print('img:',img.shape,'mask:',tmask.shape)
        tmask = np.moveaxis(tmask, 0, -1) 
        tmask=self.label2intbyonehot(tmask)
        print('img.shape:',img.shape,';tmask.shape:',tmask.shape,)
        print('dsimg.nodata:', ds.nodata,';dsmask.nodata:', dsmask.nodata)
        tmask[tmask!=key]=0
        tmask[tmask==key]=1 
        threshold = -1 
        nsample = float((patchsize + 2 * border_size) * (patchsize + 2 * border_size))
        imgref = np.moveaxis(img, 0, -1)
        print('tmask:',tmask.shape)
        tmaskref = tmask # .data
        print('tmaskref:',tmaskref.shape)
        tmaskref[((np.isnan(tmaskref)) | (tmaskref==dsmask.nodata))]=0
        ispos=np.nansum(tmaskref)>1
        tsize_r=imgref.shape[0]
        tsize_c=imgref.shape[1]
        tsize=(tsize_r,tsize_c)
        if ispos:
            rstep, cstep = oversamplingstep, oversamplingstep
            self.overSampling(img_id, threshold, ds.nodata, nsample, undersamplingstep,
                         tpath, imgref, tmaskref,tsize, border_size, patchsize, rstep, cstep) 
        return

    def label2intbyonehot(self,inmask):
        semantic_map = []
        for col in self.clsmapDf[['r','g','b']].values: 
            equality=np.equal(inmask,col)
            class_map=np.all(equality,axis=-1)
            semantic_map.append(class_map)
        semantic_map = np.stack(semantic_map,axis=-1) 
        semantic_map_index = np.argmax(semantic_map,axis=-1)
        return semantic_map_index 
   

    def samallpatchBFl(self,tbpath):
        samdsetsConfig = {}
        for key in self.clsmapDf['key']:
          if key!=0: 
            samdsetsConfig[key] = [{'scaling':1,'patchsize':self.patchsize,'border_size':self.complexbdist,
                            'oversamplingstep':self.clsmaposample[key],'undersamplingstep': 50}]
        for key, schemes in samdsetsConfig.items():
            tPath=tbpath+'/'+ str(key)
            if not os.path.exists(tPath):
                os.mkdir(tPath)
            for i in range(len(schemes)):
                tschPath=tPath+'/scheme'+str(i) 
                if os.path.exists(tschPath):
                   shutil.rmtree(tschPath, ignore_errors=True)
                os.mkdir(tschPath)
                ascheme=schemes[i]
                pd.DataFrame(ascheme,index=[key]).to_csv(tschPath+'/tscheme.csv')
                print("starting ",key," No. ",i, "... ...")
                self.startMProcess(ascheme,key, tschPath)
                print("End ",key," No. ",i,"! ")
 

    def subTrain(self,istart,iend,ascheme,key, tschPath):
        p = multiprocessing.current_process()
        print("Starting process "+p.name+", pid="+str(p.pid)+" ... ...")
        nduty=iend-istart
        for i in range(istart,iend):
            img_id=self.imgids[i]
            self.sampatchBFl(img_id, key,ascheme, tschPath)
            
    def subTrain1(self,istart,iend,ascheme, tschPath):
        p = multiprocessing.current_process()
        print("Starting process "+p.name+", pid="+str(p.pid)+" ... ...")
        nduty=iend-istart
        for i in range(istart,iend):
            img_id=self.imgids[i]
            print('running istart:',istart,'; iend:', iend,':img_id:',img_id)

    def startMProcess(self,ascheme,key, tschPath):
        n=len(self.imgids)
        nTime=int(math.floor(n/self.ncore))
        print(str(self.ncore)+" cores for "+str(n)+" duties; each core has about "+str(nTime)+" duties")
        processes = []
        for t in range(0,self.ncore):
            istart=t*nTime
            iend=(t+1)*nTime
            if t==(self.ncore-1):
                iend=n
            print('istart:',istart,'; iend:', iend)
            p = Process(name=str(t), target=self.subTrain, args=(istart, iend, ascheme, key,tschPath))
            p.daemon = True
            p.start()
            processes.append(p)
        for p in processes:
            p.join()





