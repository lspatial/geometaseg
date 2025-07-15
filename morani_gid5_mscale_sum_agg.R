

tcls=1   
rpath=paste("/geosampling/complexRSPatches/gid5_10_mcomplex_sum/",tcls,sep="")
fls=list.files(rpath) 
fls=fls[grep("^f.*\\.csv",fls)]

for(i in c(1:length(fls))){ # i=1 
  print(i)
  fullfl=paste(rpath,"/",fls[i],sep="") 
  adata=read.csv(fullfl,row.names=NULL) 
  if(i==1){
    alldata=adata
  }else{
    alldata=rbind(alldata,adata)
  }
}

plot(alldata$p5_mean,alldata$m5_mean)
 
covs=c("e21_mean","e15_mean","e11_mean","e7_mean","e5_mean")
alldata$e_all=rowMeans(alldata[,covs],na.rm=TRUE)  

covs=c("m21_mean","m15_mean","m11_mean","m7_mean","m5_mean")
alldata$m_all=rowMeans(alldata[,covs],na.rm=TRUE)  

covs=c("p21_mean","p15_mean","p11_mean","p7_mean","p5_mean")
alldata$p_all=rowMeans(alldata[,covs],na.rm=TRUE)  

hist(alldata$p_all,breaks=300) 

write.csv(alldata,paste("/geosampling/complexRSPatches/gid24_10_mcomplex_sum/patchall_cls",tcls,"_sum.csv",sep=""),row.names = FALSE)
