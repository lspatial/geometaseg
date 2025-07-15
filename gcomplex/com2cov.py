import torch
from torch import nn
from torch.nn import functional as F 
from torch_scatter import  scatter, scatter_sum 
import numpy as np 
from typing import Optional, List, Tuple, Union

def apply_along_axis(function, x, axis: int = 0, redaxis: int=0):
   return torch.stack([function(x_i,redaxis) for x_i in torch.unbind(x, dim=axis)], dim=axis)
  
def entropysamOp(p,axis=0):
    log2=2*torch.log(torch.tensor(2.0))
    ndim=p.ndim 
    cmd='p[0'
    for i in range(1,ndim):
      cmd=cmd+',:'
    cmd=cmd+']'
    p0=eval(cmd)
    cmd='p[1'
    for i in range(1,ndim):
      cmd=cmd+',:'
    cmd=cmd+']'
    p1=eval(cmd)
    orientropy=-(p0*torch.log(p0)+p1*torch.log(p1))
    res=torch.where((p1==0.0) | (p1==1.0),0,orientropy)
    res=torch.where(p1<0.5,res,log2-res)
    res=res/log2 
    return res    

class Comp2dConv(nn.Conv2d):
  
  def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        ctype: str ='entropy',
        morantype: str = 'rock', 
        device: str =None, 
        defmean: float = 0.5):
      super().__init__(in_channels,out_channels,kernel_size,stride,
            padding,dilation,groups,bias,padding_mode)
      print('init ... ...')
      self.ctype = ctype
      self.defmean = defmean 
      self.morantype = morantype
      self.W=self.gridWeiM(self.kernel_size[0],self.kernel_size[1], self.morantype,device=device)
      self.wm=torch.tensor((self.W.shape[0]+self.W.shape[1])/2)/torch.sum(self.W) 

  def gridWeiM(self,r,c,type='queen',device=None):
      d=r*c 
      w=torch.zeros((d,d),device=device)
      for i in range(d):
          for j in range(d):
              ir,ic=i//c,i%c
              jr,jc=j//c,j%c
              if ir==jr and ic==jc:
                  w[i,j]=0 
              else:
                  if type=='continuous':
                      w[i,j]=1./np.sqrt((jc-ic)*(jc-ic)+(jr-ir)*(jr-ir))
                  elif type=='rock':
                      if (jc-ic)==1 or (ic-jc)==1:
                          if (jr-ir)==1 or (ir-jr)==1: 
                              w[i,j]=1
                          elif jr==ir:
                              w[i,j]=1 
                          else:
                              w[i,j]=0 
                      elif jc==ic: 
                          if (jr-ir)==1 or (ir-jr)==1: 
                              w[i,j]=1 
                      else:
                          w[i,j]=0 
                  elif type=='queen':
                      if (jc-ic)==1 or (ic-jc)==1:
                          if (jr-ir)==1 or (ir-jr)==1: 
                              w[i,j]=0
                          elif jr==ir:
                              w[i,j]=1 
                          else:
                              w[i,j]=0 
                      elif jc==ic: 
                          if (jr-ir)==1 or (ir-jr)==1: 
                              w[i,j]=1 
                      else:
                          w[i,j]=0 
      return w 

  def forward(self, input):
      if self.ctype=='entropy':
          com=self.entropySam(input)
      elif self.ctype=='gmorani':
          com=self.gmoraniadj(input)
      elif self.ctype=='percent':
          com=self.percentage(input)          
      else:
          com=None
      return com

  def percentage(self, input):
      print('input.shape:',input.shape )
      inputdata = input.to(torch.float32) 
      bs, ch, in_h, in_w = inputdata.shape 
      out_h = (in_h + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
      out_w = (in_w + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1
      cols = F.unfold(inputdata, self.kernel_size, self.dilation, self.padding, self.stride)
      # colsV = cols.view(bs, ch, self.kernel_size[0], self.kernel_size[1], out_h, out_w)
      center_y, center_x = self.kernel_size[0] // 2, self.kernel_size[1] // 2
      print('cols:',cols.shape)
      count=torch.sum(cols,axis=1)
      deno=self.kernel_size[0]*self.kernel_size[1]
      p=torch.div(count,deno) 
#      print('p:',p.shape,'; count:',count.shape,';deno:',deno) 
      #p=torch.sum(p,axis=1)
      #print('p sum:',p.shape,'; bs:',bs) 
#      p = p[:,1,:,:]
      p = p.view(bs,out_h, out_w) 
      return p  

  def entropy(self, input):
      print('input.shape:',input.shape )
      inputdata = input.to(torch.float32) 
      bs, ch, in_h, in_w = inputdata.shape 
      out_h = (in_h + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
      out_w = (in_w + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1
      cols = F.unfold(inputdata, self.kernel_size, self.dilation, self.padding, self.stride)
      # colsV = cols.view(bs, ch, self.kernel_size[0], self.kernel_size[1], out_h, out_w)
      center_y, center_x = self.kernel_size[0] // 2, self.kernel_size[1] // 2
      ones = torch.ones(cols.shape, dtype=cols.dtype, device=cols.device)   
      print('cols.device:',cols.device)
      index=cols.long()  
      count = scatter_sum(ones, index, 1)   
      deno=self.kernel_size[0]*self.kernel_size[1]
      p=torch.div(count,deno) 
      eps=1e-8 
     # p.apply_(lambda x: (-eps*np.log(eps) if x==0 else -x*np.log(x)))  
      p.apply_(lambda x: -x*np.log(x) )  
      entropy=torch.sum(p,axis=1)
      entropyV = entropy.view(bs,out_h, out_w) 
      return entropyV  

  def entropySam(self, input):
      print('input.shape:',input.shape )
      inputdata = input.to(torch.float32) 
      bs, ch, in_h, in_w = inputdata.shape 
      out_h = (in_h + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
      out_w = (in_w + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1
      cols = F.unfold(inputdata, self.kernel_size, self.dilation, self.padding, self.stride)
      # colsV = cols.view(bs, ch, self.kernel_size[0], self.kernel_size[1], out_h, out_w)
      center_y, center_x = self.kernel_size[0] // 2, self.kernel_size[1] // 2
      ones = torch.ones(cols.shape, dtype=cols.dtype, device=cols.device)   
      print('cols.device:',cols.device)
      index=cols.long()  
      count = scatter_sum(ones, index, 1)   
      deno=self.kernel_size[0]*self.kernel_size[1]
      p=torch.div(count,deno) 
      print("p shape:",p.shape,";p:")
      entropy=apply_along_axis(entropysamOp, p, 0,0)
      entropyV = entropy.view(bs,out_h, out_w) 
      return entropyV 
    
    
  def gmorani(self,input):
      inputdata = input.to(torch.float32)  
      bs, ch, in_h, in_w = inputdata.shape 
      out_h = (in_h + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
      out_w = (in_w + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1
      cols = F.unfold(inputdata, self.kernel_size, self.dilation, self.padding, self.stride)
      # W=self.gridWeiM(self.kernel_size[0],self.kernel_size[1],'continuous',device=cols.device) 
      #W=torch.tensor(W)
      if self.W.device!=cols.device:
         self.W=self.W.to(cols.device) 
      if self.wm.device!=cols.device:
         self.wm=self.wm.to(cols.device)
      W1=self.W.expand(ch, self.W.shape[0],self.W.shape[1])
      W1=W1[0,:,:]
      cols=torch.moveaxis(cols,1,-1)
      mmean=torch.tensor(self.defmean) if self.defmean  is not None else  torch.mean(cols)
      cols=torch.tensor(cols, dtype=torch.float32)
      W1=torch.tensor(W1, dtype=torch.float32)
      mid=torch.matmul(cols-mmean,W1)
      mid=mid*(cols-mmean)
      fsum=(cols-mmean)*(cols-mmean) 
      res=torch.sum(mid, dim=-1)
      fsum=torch.sum(fsum, dim=-1)
      res= self.wm*res/fsum
      res[torch.isnan(res)] = 1 
      res = res.view(bs,out_h, out_w) 
      return res  
      
  def gmoraniadj(self,input):
      inputdata = input.to(torch.float32)  
      bs, ch, in_h, in_w = inputdata.shape 
      chn_mean=torch.mean(inputdata,dim=[2,3])
      chn_mean2=chn_mean[:,:,None,None].expand(bs, ch, in_h, in_w)
      inputdatam=inputdata-chn_mean2
      out_h = (in_h + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
      out_w = (in_w + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1
      cols = F.unfold(inputdatam, self.kernel_size, self.dilation, self.padding, self.stride)
      # W=self.gridWeiM(self.kernel_size[0],self.kernel_size[1],'continuous',device=cols.device) 
      #W=torch.tensor(W)
      if self.W.device!=cols.device:
         self.W=self.W.to(cols.device) 
      if self.wm.device!=cols.device:
         self.wm=self.wm.to(cols.device)
      W1=self.W.expand(ch, self.W.shape[0],self.W.shape[1])
      W1=W1[0,:,:]
      cols=torch.moveaxis(cols,1,-1)
      cols=torch.tensor(cols, dtype=torch.float32)
      W1=torch.tensor(W1, dtype=torch.float32)
      mid=torch.matmul(cols,W1)
      mid=mid*cols
      fsum=cols*cols 
      res=torch.sum(mid, dim=-1)
      fsum=torch.sum(fsum, dim=-1)
      res= self.wm*res/fsum 
      print('res[torch.isnan(res)]:',res[torch.isnan(res)].shape,',',res[torch.isnan(res)])
#      res[torch.isnan(res)] = 2 
      res = res.view(bs,out_h, out_w) 
      cols = F.unfold(inputdata, self.kernel_size, self.dilation, self.padding, self.stride)
      count=torch.sum(cols,axis=1)
      deno=self.kernel_size[0]*self.kernel_size[1]
      p=torch.div(count,deno) 
      p = p.view(bs,out_h, out_w)
#      print('p:',p.shape,';p:',p)
      res=torch.where(p==0.0,0,res)
      res=torch.where(p>=0.99,1,res) 
      return res  
      
