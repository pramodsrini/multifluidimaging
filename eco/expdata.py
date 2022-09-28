import numpy as np
import scipy.integrate as spi
from scipy import interpolate
from scipy import optimize
import scipy.spatial.distance as ssd
import scipy.stats as ss
from scipy.stats import pearsonr
import csv
import pylab as pltb
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpfit import mpfit
import io
import copy
#import time as t
import eco
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
currentfolder = "E:/pramod_anchroedDrop_Analysis/3_polarized_drops/1drop/DC/output"
print('\n experimental data output data at : ',currentfolder)

#### change delimiters of the text images at line 48,203

################# I. Text image data reading ###################
def readinputpar(inputfile,delimit):
    with open(inputfile+'.csv') as f:
        ncols = len(f.readline().split(','))
    data=np.loadtxt(inputfile+'.csv',delimiter=delimit,skiprows=1,usecols=range(1,ncols))
    return data

def processimage(textimage,pixel_conversion,fps,delimit): # text image to imagedata 
  print('\nfetching oscillations data from the textimage file..\n')
  def first_nonzero(arr, axis, invalid_val=-1):
    #print('\n\n\n value of arr:', arr)
    #mask = arr!=0
    mask = arr!=255 #for images of inverted colour
    #mask = arr==0 # image in not B&W (when there are problems in b&w due to holes in middle of drop)
    #print('\n\n\n value of mask:', mask)
    #print('\n\n\n type  of mask:', type(mask))
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

  def last_nonzero(arr, axis, invalid_val=-1):
    #mask = arr!=0
    mask = arr!=255 #for images of inverted colour
    #mask = arr==0 # image in not B&W (when there are problems in b&w due to holes in middle of drop)
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) #-1
    return np.where(mask.any(axis=axis), val, invalid_val)

  with open(textimage+'.csv') as f: 
        ncols = len(f.readline().split('delimit'))
  data=np.loadtxt(textimage+'.csv',delimiter=delimit,skiprows=1,usecols=range(1,ncols))
  start=0

  array_l=len(data)
  length=array_l-start

  time_length=np.shape(data)[0]
  time=np.arange(time_length)[:,np.newaxis]
  time=time/fps
  tmax=np.max(time)

  newdata=data[start:length+start,:]

  value_left= first_nonzero(newdata, axis=1, invalid_val=-1)[:,np.newaxis]
  value_right= last_nonzero(newdata, axis=1, invalid_val=-1)[:,np.newaxis]
  value_left=value_left/pixel_conversion
  value_right=value_right/pixel_conversion ### edited
  p=np.where(time<tmax)

  t=time[p][:,np.newaxis]
  v_1=value_left[p][:,np.newaxis]
  v_2=value_right[p][:,np.newaxis]

  v_1=v_1-v_1[0]    #shifting left drop displacement reference
  v_2=v_2[0]-v_2    #shifting right drop displacement reference
  t_start = np.argmax(v_1!=0)   # start time reference
  #print ('\n\n\n processing image.. \n tstart of drop movement= ',t_start)
  v_left=v_1[t_start:]    # date only after start time
  v_right=v_2[t_start:]
  t_shifted = t-t[t_start] #shift t reference
  t_shifted = t[t_start:]   # remove t before t=zero
  dataout=np.hstack((t_shifted,v_left,v_right))
  return (dataout)       # returns imagedata class object

def getcsvdata(filename,delimit) :          #self attribute of imagedata class object. additional/redundant function
  datafile = np.genfromtxt(filename+'.csv', delimiter=delimit,skip_header=1)#, names=True,dtype=None)
  tdata=datafile[:,0]
  x1data=datafile[:,1]
  x2data=datafile[:,2]
  stoptime = tdata[len(tdata)-1]
  numpoints = len(tdata)
  return (tdata,x1data,x2data,stoptime,numpoints) 

def pxlresolve(tdata,x1data,x2data): #due to image resolution multiple consequenct time steps indicate the same pixel position 
  print('\nresolving the low quality oscillations data from images...\n')
  t1contr =[]
  t2contr =[]
  x1contr =[]
  x2contr = []
  t1contr1 =[]
  t2contr1 =[]
  x1contr1 =[]
  x2contr1 = []
  tstart = tdata[0]
  tstep = tdata[1]-tdata[0]
  tend = tdata[len(tdata)-1]
  numpoints = len(tdata)
  j=0
  for i in range(1,numpoints-1):
    if (x1data[i]!=x1data[i-1] or x1data[i]!=x1data[i+1]): # and (x1contr1=[] or x1data[i]!=x1contr1[-1])
        t1contr1.append(tdata[i])
        x1contr1.append(x1data[i])
        j=j+1
  for i in range (0,j-1):
    t1contr.append((t1contr1[i]+t1contr1[i+1])/2)
    x1contr.append((x1contr1[i]+x1contr1[i+1])/2)
  l=0
  for i in range(1,numpoints-1):
    if x2data[i]!=x2data[i-1] or x2data[i]!=x2data[i+1]:
        t2contr1.append(tdata[i])
        x2contr1.append(x2data[i])
        l=l+1
  for i in range (0,l-1):
    t2contr.append((t2contr1[i]+t2contr1[i+1])/2)
    x2contr.append((x2contr1[i]+x2contr1[i+1])/2)
  return(t1contr,x1contr,t2contr,x2contr)

def pxlresolve_imgdata(imgdata): #due to image resolution multiple consequent time steps indicate the same pixel position 
  t1contr =[]
  t2contr =[]
  x1contr =[]
  x2contr = []
  t1contr1 =[]
  t2contr1 =[]
  x1contr1 =[]
  x2contr1 = []
  #tdata=imgdata[:,0]
  tdata=imgdata.tdata
  #x1data=imgdata[:,1]
  x1data=imgdata.x1data
  #x2data=imgdata[:,2]
  x2data=imgdata.x2data
  tstart = tdata[0]
  tstep = tdata[1]-tdata[0]
  tend = tdata[len(tdata)-1]
  numpoints = len(tdata)
  j=0
  for i in range(1,numpoints-1):
    if x1data[i]!=x1data[i-1] or x1data[i]!=x1data[i+1]:
        t1contr1.append(tdata[i])
        x1contr1.append(x1data[i])
        j=j+1
  for i in range (0,j-1):
    t1contr.append((t1contr1[i]+t1contr1[i+1])/2)
    x1contr.append((x1contr1[i]+x1contr1[i+1])/2)
  l=0
  for i in range(1,numpoints-1):
    if x2data[i]!=x2data[i-1] or x2data[i]!=x2data[i+1]:
        t2contr1.append(tdata[i])
        x2contr1.append(x2data[i])
        l=l+1
  for i in range (0,l-1):
    t2contr.append((t2contr1[i]+t2contr1[i+1])/2)
    x2contr.append((x2contr1[i]+x2contr1[i+1])/2)
  return(t1contr,x1contr,t2contr,x2contr)

class imagedata(object):              #take raw csv input of text images;shared func attrb of plotting, data saving; self: data
  """image data class objects inherits methods and data of processing imageJ raw csv data to segregate and plot"""
  def __init__(self, filename,pixel_conversion,fps,delimit):
    self.filename = filename
    self.pixel_conversion = pixel_conversion
    self.fps = fps
    #self.rawdata=([])
    print('\nopening textimage csv file\n')
    with open(self.filename+'.csv') as f:
        ncols = len(f.readline().split('delimit'))
        #tabdelimit_textimg=np.loadtxt((x.replace(',','\t') for x in f))
    #self.rawdata = np.genfromtxt(filename+'.csv', delimiter=',',skip_header=1)
    
    #s = open(self.filename+'.csv').read().replace(',','\t')
    #data = np.loadtxt(io.StringIO(s),skiprows=1,dtype=str)
    #self.rawdata =np.loadtxt(self.filename+'.csv',delimiter=delimit,skiprows=1,usecols=range(1,ncols))
    self.datafile = processimage(self.filename,self.pixel_conversion,self.fps,delimit)
    self.tdata=self.datafile[:,0]
    self.x1data=self.datafile[:,1]
    self.x2data=self.datafile[:,2]
    self.tstart = self.tdata[0]
    self.tend = self.tdata[len(self.tdata)-1]
    self.numpoints = len(self.tdata)
    self.resolimage =pxlresolve(self.tdata,self.x1data,self.x2data)
    (self.t1resol,self.x1resol,self.t2resol,self.x2resol) = self.resolimage
    #(self.v1data,self.v2data,self.a1data,self.a2data,self.theta1,self.theta2,self.x1_theta,self.x2_theta)=data_derivative_par(self.resolimage)

  def writedata(self): 
    filespath = os.path.join(currentfolder, self.filename)
    if not os.path.exists(filespath):
      os.mkdir(filespath)
    output_file=os.path.join(filespath,'out_'+self.filename+'.csv')
    #output_file=currentfolder+'/'+self.filename+'/'+'out_'+self.filename+'.csv'
    np.savetxt(output_file,self.datafile)

  def plttimeseries (self): 
    fig = plt.figure(figsize=(10, 5))
    gs = GridSpec(nrows=2, ncols=1)
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(self.tdata,self.x1data,'-')
    ax0.set_title('x1') 
    ax0.legend(['expt'])

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(self.tdata,self.x2data,'-')
    ax1.set_title('x2')
    ax1.legend(['expt'])

    plt.suptitle ('x1 and x2 of experimental data')
    plt.draw()

    filespath = os.path.join(currentfolder, self.filename)
    if not os.path.exists(filespath):
      os.makedirs(filespath)
    plt.savefig(currentfolder+'/'+self.filename+'/'+self.filename+'_tseries_exp.png')
    plt.show()

  def fft(self):
    self.yf1=np.fft.fft(self.x1data)
    self.yf2=np.fft.fft(self.x2data)
    #freq=1/(data[:,0])
    self.del_t = self.tdata[1] - self.tdata[0]  # sampling interval i.e., time step size
    #N = len(tdata)
    self.freq = np.linspace(0, 1 / self.del_t, self.numpoints)

    plt.plot(self.freq, np.abs(self.yf1),'.b',self.freq,np.abs(self.yf2),'*g')
    plt.legend(['drop1','drop2'])

    self.y1=np.abs(self.yf1)
    self.y2=np.abs(self.yf1)
    self.y1=self.y1[:,np.newaxis]
    self.y2=self.y2[:,np.newaxis]
    self.freq=self.freq[:,np.newaxis]
    out=np.hstack((self.freq,self.y1,self.y2))
    np.savetxt('fft_400V_x2.txt',out)
    plt.show()

################# II. oscillations data processing ###################

def data_derivative_par(resolimage):
  print('\nEstimating velocity, acceleration and phase angle of oscillations...\n')
  t1data=resolimage[0]
  x1data=resolimage[1]
  t2data=resolimage[2]
  x2data=resolimage[3]
  theta1=[0]
  theta2=[0]
  x1_theta=[0]
  x2_theta=[0]
  totalpoints1_in_each_cycle=[]
  totalpoints2_in_each_cycle=[]
  v1data=[]
  v2data=[]
  a1data = []
  a2data = []
  interiorpoint1_count=0
  interiorpoint2_count=0
  numpoints1 = len(t1data)
  numpoints2 = len(t2data)
  index1 =0
  index2 =0

  ###velocity and acceleration at different times
  for i in np.arange(1,numpoints1-1): #central diff for velocity
    v1data.append((x1data[i+1]-x1data[i-1])/(t1data[i+1]-t1data[i-1]))
  for i in np.arange(1,numpoints2-1):
    v2data.append((x2data[i+1]-x2data[i-1])/(t2data[i+1]-t2data[i-1]))
  for i in np.arange(2,numpoints1-3): #central diff for acceleration
    a1data.append((v1data[i+1]-v1data[i-1])/(t1data[i+1]-t1data[i-1]))
  for i in np.arange(2,numpoints2-3):
    a2data.append((v2data[i+1]-v2data[i-1])/(t2data[i+1]-t2data[i-1]))

  ###phase angle and corresponding dispacements
  for i in np.arange(1,numpoints1-1):
    if x1data[i]>x1data[i-1] and x1data[i]>x1data[i+1]:
      totalpoints1_in_each_cycle.append(interiorpoint1_count+1)
      #print(interiorpoint1_count+1)  
      interiorpoint1_count=0   
      #print('Thisssss :',len(totalpoints1_in_each_cycle),'\n\n')
    else:
      interiorpoint1_count = interiorpoint1_count+1

  #print(len(totalpoints1_in_each_cycle[0]))
  for i in np.arange(0,totalpoints1_in_each_cycle[0]-1):
    index1 = index1+1
    theta1.append(i*np.pi/totalpoints1_in_each_cycle[0]-0.5*np.pi)
    x1_theta.append(x1data[index1])
  #index1 = len(x1_theta)-1
  #index1 = totalpoints1_in_each_cycle[0]-1
  for j in np.arange(1,len(totalpoints1_in_each_cycle)): # cycle by cycle
    for i in np.arange(0,totalpoints1_in_each_cycle[j]-1): # point by point in each cycle
      index1 = index1 + 1
      theta1.append( i*2*np.pi/totalpoints1_in_each_cycle[j]+np.pi/2)
      x1_theta.append(x1data[index1])
      #x1_theta.append(index1)

  for i in np.arange(1,numpoints2-1):
    if x2data[i]>x2data[i-1] and x2data[i]>x2data[i+1]:
      totalpoints2_in_each_cycle.append(interiorpoint2_count+1)
      interiorpoint2_count=0      
    else:
      interiorpoint2_count = interiorpoint2_count+1

  for i in np.arange(0,totalpoints2_in_each_cycle[0]-1):
    index2 = index2+1
    theta2.append(i*np.pi/totalpoints2_in_each_cycle[0]-0.5*np.pi)
    x2_theta.append(x2data[index2])
    #theta2.append(0)
    #x2_theta.append(0)
  #index2 = len(x2_theta)-1
  #index2 = totalpoints2_in_each_cycle[0]-1
  for j in np.arange(1,len(totalpoints2_in_each_cycle)):
    for i in np.arange(0,totalpoints2_in_each_cycle[j]-1):
      index2 = index2 + 1
      theta2.append( i*2*np.pi/totalpoints2_in_each_cycle[j]+np.pi/2)
      x2_theta.append(x2data[index2])
      #x2_theta.append(index2)

  return (v1data,v2data,a1data,a2data,theta1,theta2,x1_theta,x2_theta)

def findenvelop(resolvedimgdata):
  print('\nresolving the maximum and minimum displacement data points of the oscilaltions...\n')
  x1max=[]
  x1min=[]
  x2max=[]
  x2min=[]
  t1max=[]
  t1min=[]
  t2max=[]
  t2min=[]
  tosc1=[]
  tosc2=[]
  x1qs =[]
  x2qs =[]
  Tosc1=[]
  Tosc2=[]
  #theta1=[0]
  #theta2=[0]
  #x1_theta=[0]
  #x2_theta=[0]
  totalpoints1_incycle=[]
  totalpoints2_incycle=[]
  #v1data=[]
  #v2data=[]
  #a1data = []
  #a2data = []
  #print(type(x1max))
  t1data=resolvedimgdata[0]
  x1data=resolvedimgdata[1]
  t2data =resolvedimgdata[2]
  x2data=resolvedimgdata[3]
  t1start = t1data[0]
  t1step = t1data[1]-t1data[0]
  t1end = t1data[len(t1data)-1]
  numpoints1 = len(t1data)
  numpoints2 = len(t2data)
  interiorpoint1_count=0
  interiorpoint2_count=0

  """
  for i in np.arange(1,numpoints1-1): #BDF for velocity
    v1data.append(x1data[i]-x1data[i-1])
  for i in np.arange(1,numpoints2-1):
    v2data.append(x2data[i]-x2data[i-1])
  for i in np.arange(2,numpoints1-2): #BDF for acceleration
    a1data.append(v1data[i]-v1data[i-1])
  for i in np.arange(2,numpoints2-2):
    a2data.append(v2data[i]-v2data[i-1])
  """
  """"
  if x1data[0]<x1data[1]: to include t=0 point in the envelop
    x1min.append(x1data[0])
    t1min.append(t1data[0])
  elif x1data[0]>x1data[1]:
    x1max.append(x1data[0])
    t1max.append(t1data[0])
  """
  if x1data[0]<x1data[1]:
    x1min.append(x1data[0])
    t1min.append(t1data[0])

  if x1data[0]>x1data[1]:
    x1max.append(x1data[0])
    t1max.append(t1data[0])

  for i in np.arange(1,numpoints1-1):
    if x1data[i]>x1data[i-1] and x1data[i]>x1data[i+1]:
      x1max.append(x1data[i])
      t1max.append(t1data[i])
      totalpoints1_incycle.append(interiorpoint1_count+1)
      interiorpoint1_count=0      

    elif x1data[i]<x1data[i-1] and x1data[i]<x1data[i+1]:
      x1min.append(x1data[i])
      t1min.append(t1data[i])
      
    else:
      interiorpoint1_count = interiorpoint1_count+1
  """
  for i in np.arange(0,totalpoints1_incycle[0]-1):
    theta1.append(0)
    x1_theta.append(0)
  index1 = totalpoints1_incycle[1]-1
  for j in np.arange(1,len(totalpoints1_incycle)):
    for i in np.arange(0,totalpoints1_incycle[j]-1):
      index1 = index1 + i
      theta1.append( i*2*np.pi/totalpoints1_incycle[j]+np.pi/2)
      x1_theta.append(index1)
  """
  if x2data[0]<x2data[1]:
    x2min.append(x2data[0])
    t2min.append(t2data[0])

  if x2data[0]>x2data[1]:
    x2max.append(x2data[0])
    t2max.append(t2data[0])

  for i in np.arange(1,numpoints2-1):
    if x2data[i]>x2data[i-1] and x2data[i]>x2data[i+1]:
      x2max.append(x2data[i])
      t2max.append(t2data[i])
      totalpoints2_incycle.append(interiorpoint2_count+1)
      interiorpoint2_count=0  
    elif x2data[i]<x2data[i-1] and x2data[i]<x2data[i+1]:
      x2min.append(x2data[i])
      t2min.append(t2data[i])
    else:
      interiorpoint2_count = interiorpoint2_count+1
  """
  for i in np.arange(0,totalpoints2_incycle[0]-1):
    theta2.append(0)
    x2_theta.append(0)  
  index2 = totalpoints2_incycle[1]-1
  for j in np.arange(0,len(totalpoints2_incycle)):
    for i in np.arange(0,totalpoints2_incycle[j]-1):
      index2 = index2 + i
      theta2.append( i*2*np.pi/totalpoints2_incycle[j]+np.pi/2)
  """
  #print(len(t1max))
  #print(len(t1min))
  #print(len(t2max))
  #print(len(t2min))

  # quasistatic estimate fit
  print('\n finding average parameters of each cycle...\n')
  for i in range(0,min(len(t1max),len(t1min))-2):  # -1 should suffice; but error in one case #min(len(t1max),len(t2max))
    tosc1.append((t1max[i]+t1min[i])/2)
    x1qs.append((x1max[i]+x1min[i])/2)
  
  if len(tosc1)==2:
    Tosc1_avg = tosc1[1]-tosc1[0]
  elif len(tosc1)<2:
    Tosc1_avg = 1e6
  else:
    for i in range(1,len(tosc1)-1):     # duration of each cycle
      Tosc1.append(tosc1[i]-tosc1[i-1])
    Tosc1_avg = sum(Tosc1)/(len(Tosc1))
    

  for i in range(0,min(len(t2max),len(t2min))-2):
    tosc2.append((t2max[i]+t2min[i])/2)
    x2qs.append((x2max[i]+x2min[i])/2)
  if len(tosc2)==2:
    Tosc2_avg = tosc2[1]-tosc2[0]
  elif len(tosc2)<2:
    Tosc2_avg = 1e6
  else:
    for i in range(1,len(tosc2)-1):
      Tosc2.append(tosc2[i]-tosc2[i-1])
    Tosc2_avg = sum(Tosc2)/(len(Tosc2))
  #print('\n\n\n\nTosc2 :',Tosc2)
  #print('\n\n\n\n\n\n')
  return (t1max,x1max,t1min,x1min,t2max,x2max,t2min,x2min,Tosc1,Tosc2,Tosc1_avg,Tosc2_avg,x1qs,x2qs)#,v1data,v2data,a1data,a2data,theta1,theta2,x1_theta,x2_theta)

#def cyclewisedata(resolvedimgdata)

class envelopdata(object):
  """max/min envelop data from experiment- imagedata object"""
  def __init__(self, imagedata):
    self.envdata= findenvelop(imagedata.resolimage)
    self.x1max=self.envdata[1]
    self.x1min=self.envdata[3]
    self.x2max=self.envdata[5]
    self.x2min=self.envdata[7]
    self.t1max=self.envdata[0]
    self.t1min=self.envdata[2]
    self.t2max=self.envdata[4]
    self.t2min=self.envdata[6]
    self.Tosc1=self.envdata[8]
    self.Tosc2=self.envdata[9]
    self.Tosc1_avg=np.asarray(self.envdata[10])
    self.Tosc2_avg=np.asarray(self.envdata[11])
    self.x1qs=self.envdata[12]
    self.x2qs=self.envdata[13]

    #self.fit_env_2min=eco.envelopfit_min(self.t2min,self.x2min) # shifted to eco.envfit
    #self.fit_env_2max=eco.envelopfit_max(self.t2max,self.x2max)
    #self.fit_env_1min=eco.envelopfit_min(self.t1min,self.x1min)
    #self.fit_env_1max=eco.envelopfit_max(self.t1max,self.x1max)

    #self.dynamicphase=eco.finddynphase(imagedata,self.t1max,self.t1min,self.t2max,self.t2min,self.x1max,self.x1min,self.x2max,self.x2min,coef)
    #(self.dynphase, self.ratio1,self.ratio2,self.xaxis,self.yaxis,self.omg_osc1_avg,self.omg_osc2_avg,self.omg_b1,self.omg_b2)= self.dynamicphase
    
    """
    self.v1data=self.envdata[14]
    self.v2data=self.envdata[15]
    self.a1data=self.envdata[16]
    self.a2data=self.envdata[17]
    self.theta1_data=self.envdata[18]
    self.theta2_data=self.envdata[19]
    self.x1_theta=self.envdata[20]
    self.x2_theta=self.envdata[21]
    """

  def writedata(self): 
    filespath = os.path.join(currentfolder, self.filename)
    if not os.path.exists(filespath):
      os.mkdir(filespath)
    output_file=os.path.join(filespath,'out_'+self.filename+'.csv')
    #output_file=currentfolder+'/'+self.filename+'/'+'out_'+self.filename+'.csv'
    np.savetxt(output_file,self.datafile)

  def plttimeseries (self): 
    fig = plt.figure(figsize=(10, 5))
    gs = GridSpec(nrows=2, ncols=1)
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(self.tdata,self.x1data,'-')
    ax0.set_title('x1') 
    ax0.legend(['expt'])

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(self.tdata,self.x2data,'-')
    ax1.set_title('x2')
    ax1.legend(['expt'])

    plt.suptitle ('x1 and x2 of experimental data')
    plt.draw()

    filespath = os.path.join(currentfolder, self.filename)
    if not os.path.exists(filespath):
      os.mkdirs(filespath)
    plt.savefig(currentfolder+'/'+self.filename+'/'+self.filename+'_tseries_exp.png')
    plt.show()

  def fft(self):
    self.yf1=np.fft.fft(self.x1data)
    self.yf2=np.fft.fft(self.x2data)
    #freq=1/(data[:,0])
    self.del_t = self.tdata[1] - self.tdata[0]  # sampling interval i.e., time step size
    #N = len(tdata)
    self.freq = np.linspace(0, 1 / self.del_t, self.numpoints)

    plt.plot(self.freq, np.abs(self.yf1),'.',self.freq,np.abs(self.yf2),'-')
    plt.legend(['drop1','drop2'])

    self.y1=np.abs(self.yf1)
    self.y2=np.abs(self.yf1)
    self.y1=self.y1[:,np.newaxis]
    self.y2=self.y2[:,np.newaxis]
    self.freq=self.freq[:,np.newaxis]
    out=np.hstack((self.freq,self.y1,self.y2))
    np.savetxt('fft_400V_x2.txt',out)
    plt.show()