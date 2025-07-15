
tfl='/geosampling/pretrained_2/meta_all_0.2/SwinUNet_Model_test1_0.2/hist_meta_train.csv'
metamodel=read.csv(tfl,row.names = NULL) 
mm=max(metamodel$test_miou);max(metamodel$test_iou_1)
metamodel[which(metamodel$test_miou==mm),]


tfl='/aiprof/segform_pretrained/seg_finetune_decoder_1_4/hist_segformer.csv'
segform_decoder=read.csv(tfl,row.names = NULL) 
mm=max(segform_decoder$test_miou);max(segform_decoder$test_iou_1)
segform_decoder[which(segform_decoder$test_miou==mm),]

tfl='/aiprof/segform_pretrained/seg_finetune_all_2_2/hist_segformer.csv'
segform_full=read.csv(tfl,row.names = NULL) 
mm=max(segform_full$test_miou);max(segform_full$test_iou_1)
segform_full[which(segform_full$test_miou==mm),]

tfl='/geosampling/pretrained/swinunet/SwinUNet_Model_test1_0.1/hist_swinunet.csv'
swinunet_full=read.csv(tfl,row.names = NULL) 
mm=max(swinunet_full$test_miou);max(swinunet_full$test_iou_1)
swinunet_full[which(swinunet_full$test_miou==mm),]

tfl='/geosampling/pretrained_3/unet_full/SwinUNet_Model_test1_0.1/hist_unet.csv'
baselineunet_full=read.csv(tfl,row.names = NULL) 
mm=max(baselineunet_full$test_miou);max(baselineunet_full$test_iou_1)
baselineunet_full[which(baselineunet_full$test_miou==mm),]

tfl='/geosampling/pretrained_3/metamcls_seg_full/SwinUNet_Model_test1_0.1/hist_meta_train_mcls.csv'
metacls_full=read.csv(tfl,row.names = NULL) 


target='test_miou'
par(mar=c(4,4,1,1),mfrow=c(1,2))  
plot(metamodel$epoch,metamodel[,target],type='l',col='red',
     ylim=c(0.4,0.9),xlim=c(0,90),lwd=2,xlab="Epoch",ylab="Test MIoU") 
lines(segform_decoder$epoch,segform_decoder[,target],col='blue',lwd=2) 
lines(segform_full$epoch,segform_full[,target],col='black',lwd=2) 
lines(baselineunet_full$epoch,baselineunet_full[,target],col='darkgray',lwd=2) 
legend(-5,0.55,lty=1,lwd=2,col=c("red","blue","black","darkgray"),seg.len=0.3,x.intersp=0.2,y.intersp=1.2,
       legend=c("Fully finetuning meta UNet","Finetuning decoders of pretrained SegFormer",
                "Fully finetuning pretrained SegFormer","UNet"),bty="n",cex=0.8)  

target='test_iou_1'
plot(metamodel$epoch,metamodel[,target],type='l',col='red',ylim=c(0.4,0.9),xlim=c(0,90),lwd=2,
     xlab="Epoch",ylab="Test IoU of build-ups") 
lines(segform_decoder$epoch,segform_decoder[,target],col='blue',lwd=2) 
lines(segform_full$epoch,segform_full[,target],col='black',lwd=2) 
lines(baselineunet_full$epoch,baselineunet_full[,target],col='darkgray',lwd=2)  
legend(-5,0.55,lty=1,lwd=2,col=c("red","blue","black","darkgray"),seg.len=0.3,x.intersp=0.2,y.intersp=1.2,
       legend=c("Fully finetuning meta UNet","Finetuning decoders of pretrained SegFormer",
                "Fully finetuning pretrained SegFormer","UNet"),bty="n",cex=0.8) 


target='train_miou'
par(mar=c(4,4,1,1),mfrow=c(1,2))  
plot(metamodel$epoch,metamodel[,target],type='l',col='red',
     ylim=c(0.4,1),xlim=c(0,90),lwd=2,xlab="Epoch",ylab="Train MIoU") 
lines(segform_decoder$epoch,segform_decoder[,target],col='blue',lwd=2) 
lines(segform_full$epoch,segform_full[,target],col='black',lwd=2) 
lines(baselineunet_full$epoch,baselineunet_full[,target],col='darkgray',lwd=2) 
legend(-5,0.55,lty=1,lwd=2,col=c("red","blue","black","darkgray"),seg.len=0.3,x.intersp=0.2,y.intersp=1.2,
       legend=c("Fully finetuning meta UNet","Finetuning decoders of pretrained SegFormer",
                "Fully finetuning pretrained SegFormer","UNet"),bty="n",cex=0.8)  

target='train_iou_1'
plot(metamodel$epoch,metamodel[,target],type='l',col='red',ylim=c(0.4,1),xlim=c(0,90),lwd=2,
     xlab="Epoch",ylab="Train IoU of build-ups") 
lines(segform_decoder$epoch,segform_decoder[,target],col='blue',lwd=2) 
lines(segform_full$epoch,segform_full[,target],col='black',lwd=2) 
lines(baselineunet_full$epoch,baselineunet_full[,target],col='darkgray',lwd=2)  
legend(-5,0.55,lty=1,lwd=2,col=c("red","blue","black","darkgray"),seg.len=0.3,x.intersp=0.2,y.intersp=1.2,
       legend=c("Fully finetuning meta UNet","Finetuning decoders of pretrained SegFormer",
                "Fully finetuning pretrained SegFormer","UNet"),bty="n",cex=0.8) 

# lines(metacls_full$epoch,metacls_full[,target],col='pink',lwd=2)  

