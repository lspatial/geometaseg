import pandas as pd 
import numpy as np
import os
import pickle


def samplehelper(sampleDf_,uid,trainprp_,testprp_=None,testsamplestype="random",stratify=None,weight=None):
    if stratify is None:
        wei=sampleDf_[weight] if weight is not None else None 
        trainsampleDf=sampleDf_.sample(frac=trainprp_,weights=wei)
        if testprp_ is None: 
           return trainsampleDf,None 
        else:
           restsamples=sampleDf_.loc[~sampleDf_[uid].isin(trainsampleDf[uid].values)] 
           testp_adj=testprp_*len(sampleDf_)/len(restsamples)
           testp_adj=1 if testp_adj>1 else testp_adj
           wei=restsamples[weight] if weight is not None and testsamplestype in ['onlywei','train'] else None 
           testsampleDf=restsamples.sample(frac=testp_adj,weights=wei)
           return trainsampleDf,testsampleDf 
    wei=sampleDf_[weight] if weight is not None else None
    trainsampleDf=sampleDf_.groupby(stratify).sample(frac=trainprp_,weights=wei) 
    if testprp_ is None: 
       return trainsampleDf,None  
    restsamples=sampleDf_.loc[~sampleDf_[uid].isin(trainsampleDf[uid].values)] 
    testp_adj=testprp_*len(sampleDf_)/len(restsamples)
    testp_adj=1 if testp_adj>1 else testp_adj
    if testsamplestype=='random':
       testsampleDf=restsamples.sample(frac=testp_adj)
    elif testsamplestype=='train':
       wei=restsamples[weight] if weight is not None else None 
       testsampleDf=restsamples.groupby(stratify).sample(frac=testp_adj,weights=wei)
    elif testsamplestype=='onlywei':
       wei=restsamples[weight] if weight is not None else None 
       testsampleDf=restsamples.sample(frac=testp_adj,weights=wei)
    elif testsamplestype=='onlystratify':
       testsampleDf=restsamples.groupby(stratify).sample(frac=testp_adj)
    else:
       print("Please provide valid testsamplestype!")
    return trainsampleDf,testsampleDf 


def sampling(sampleSFile,uid,trainprp, testprp, testsamplestype="random", stratify=None, strange=None, weight=None):
    """Split the samples based on the sample statistical file and requirements.
    
        Args:
            sampleSFile (str): the statistical CSV file of patch samples; 
            uid (str): unique column name for sample id; 
            trainprp (float): sampling proportion for training;
            testprp (float or str): sampling proportion for testing samples or data frame file for recording the test index files;
            testsamplestype (str): sampling method for test samples, "train": using similar method as training samples; "random": random sapling;
            stratify (str): stratification variable name to be used in the sample statistical file, None means no stratifications;
            strange (int,list): the number of stratra if strange is an integer, or a quantile range if it is a list; 
            weight (str): weighing variable name to be used in the sample statistical file, None means no weighing.  

        Returns:
           Two data frames for training and testing  
    """
    assert  sampleSFile is not None, "please provide the sample statistical file!"
    sampleSDF=pd.read_csv(sampleSFile)
    if not uid in sampleSDF.columns:
        sampleSDF[uid] = np.array([i for i in range(1, sampleSDF.shape[0] + 1)])
    assert not sampleSDF[uid].duplicated().any(), "Duplicated sample ID is not allowed, please ensure unique sample id!"
    testsamples=None 
    if isinstance(testprp,str):
        assert os.path.exists(testprp), "Please provide the correct file of test sample index!"
        with open(testprp, 'rb') as fp:   #Pickling
          testsamples=pickle.load(fp)
    if stratify is None: 
        if testsamples is not None: 
           restsampleSDF=sampleSDF.loc[~sampleSDF[uid].isin(testsamples[uid])] 
           trainp_adj=trainprp*len(sampleSDF)/len(restsampleSDF)
           trainp_adj=1 if trainp_adj>1 else 1 
           trainsamples,_=samplehelper(restsampleSDF,uid,trainp_adj,None,None,None,weight)
           return trainsamples,testsamples 
        else: 
           return samplehelper(sampleSDF,uid,trainprp,testprp,testsamplestype,None,weight)
    if isinstance(strange,list):
       quantile=strange 
    elif isinstance(strange,int):
       quantile=[1.0/float(strange)*i for i in range(strange+1)]
    else:
       raise Exception("Please provide the correct type of stratra information!")
    labels = ['q'+str(i) for i,k in enumerate(quantile) if i>0]
    sampleSDF[sampleSDF[stratify]<0]=0
    sampleSDF[sampleSDF[stratify]>1]=1
    sampleSDF['quantile'] = pd.qcut(sampleSDF[stratify], q = quantile, labels = labels)
    if testsamples is not None: 
       restsampleSDF=sampleSDF.loc[~sampleSDF[uid].isin(testsamples[uid])] 
       trainp_adj=trainprp*len(sampleSDF)/len(restsampleSDF)
       trainp_adj=1 if trainp_adj>1 else 1 
       trainsamples,_=samplehelper(restsampleSDF,uid,trainp_adj,None,None,'quantile',weight)
       return trainsamples,testsamples 
    return samplehelper(sampleSDF,uid,trainprp,testprp,testsamplestype,'quantile',weight)


def saveSamples(seltrain,seltest,rpath='/tmp'):
    tfl=rpath+'/seltrain.pkl'
    with open(tfl, 'wb') as fp:   #Pickling
      pickle.dump(seltrain, fp)
    tfl=rpath+'/seltest.pkl'
    with open(tfl, 'wb') as fp:   #Pickling
      pickle.dump(seltest, fp)






