import numpy as np
#from curve_fit import annealing
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
import copy
import eco
#import time as t
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
currentfolder = os.getcwd()
print('\n input files are from : ',currentfolder)

####################################################################
#################1. import experiment data #########################
###a.writing all file names to a list- filenames; and all other data to a list of strings- imgpar
filenames=[]
with open('index_1drop.csv', newline='') as csvfile:
	reader = csv.reader(csvfile,delimiter='\t')
	for row in reader:
		filenames.append(row[0])
#print('\n\n',filenames[0])
#print('\n\n', row)

with open('index_1drop.csv') as f:
	ncols = len(f.readline().split('\t')) #ensure the delimiter of the inputs csv file
imgpar=np.genfromtxt('index_1drop.csv',delimiter='\t',skip_header=1,usecols=range(2,ncols), dtype=str)
#imgpar=np.loadtxt('inputfile_DCdamping_twodrop.csv',delimiter='\t',skiprows=1,usecols=range(2,ncols))


###b. processing each image in the master loop
for filenum in range(0,len(imgpar)): 
	filename = filenames[filenum+1]
	#print(imgpar[filenum])

	#filespath = os.path.join(currentfolder, textimages_twodrops)
	
###c. writing data of current case of the loop into temporary variables: 'par'-values of input parameter; and 'coef'-values of guess coefficients
	exp_input= np.delete(imgpar[filenum],20).astype(np.float) #All input parameters except the string of DC or AC are converted into float- exp_input
	#exp_input= imgpar[filenum]
	print('starting case #',filenum+1,':\n \t name :',filename)
	par= (exp_input[0],exp_input[1],exp_input[2],exp_input[3],exp_input[4],exp_input[5],exp_input[6],exp_input[7],exp_input[8],exp_input[9],exp_input[10],exp_input[11],imgpar[filenum][20])
	(fps,pxl_resol,radius1,radius2,density_drop,visc_drop,density_bulk,visc_bulk,permitivity_bulk,ift,V,s0,f_ac) =par
	coef= [exp_input[13],exp_input[14],exp_input[15],exp_input[16],exp_input[17],exp_input[18],s0]
	print('list of par:  ', par)
	print('list of coeff:  ', coef,'\n')

	
	if exp_input[19]==1.0:  #receive the value of delimiter in the textimages file
		delimit = '\t'
	elif exp_input[19]==0.0:
		delimit = ','
	else:
		print('input delimiter value invalid\n 0 = comma \n 1 = tab ')
	print('\n\n\n\n delimiter of this text image input :  ', delimit)

	#x0 = [0,0,0,0]

	######### I. Image processing ##########
	print('\n>>>>>>>>>>>> Initiating image processing <<<<<<<<<<\n')
	img=eco.imagedata(currentfolder+'/'+filename,pxl_resol,fps,delimit)
	# x0=[0,0,0,img.x2data[-1]]
	#img.x2data=[i * -1 for i in img.x2data] #relaxation : delete this ?
	#(img.x2resol, img.x2data, img.v2data, img.a2data) = ([i * -1 for i in img.x2resol], [i * -1 for i in img.x2data], [i * -1 for i in img.v2data], [i * -1 for i in img.a2data]) #for compressing drop (along common axis)
	#imgenv=eco.envelopdata(img)
	#print(imgenv.t2min[0],imgenv.x2min[0])

	# x0=[0,0,0,10*imgenv.x2min[0]/imgenv.t2min[0]] #relaxation case
	#print(imgenv.t1min)
#	v_scale =(img.v1data[0]+img.v2data[0])/2
	#delm=eco.write_ecopar(par,img,imgenv,filename) #all basic data from images written to a csv file.
#	eco.write_ehdpar(par,v_scale,filename) #EHD parameters written to a csv

	np.savetxt(currentfolder+'/'+'Time_series'+'/'+filename+'.csv', np.c_[img.tdata,img.x1data,img.x2data], delimiter=',') #Time series file

	eco.plttimeseries (img.tdata,img.x1data,img.tdata,img.x2data,filename)	
	eco.plttimeseries_yscaled (img.tdata,img.x1data,img.tdata,img.x2data,filename,s0,radius1,radius2)
	eco.plttimeseries_g_h_scaled (img.tdata,img.x1data,img.x2data,filename,s0,radius1,radius2)
	eco.phaseplot1(img.x1data,img.x2data,filename)
	#eco.phaseplot2(img.x1resol,img.x2resol,img.v1data,img.v2data,filename)
	eco.fft(img.x1data,img.x2data,img.tdata,filename)
	#eco.polarplot(img.theta1,img.theta2,img.x1_theta,img.x2_theta,filename)
	#eco.kineticsplot1(img.x1resol,img.x2resol,img.a1data,img.a2data,1,delm,filename)
	#eco.kineticsplot2(img.x1resol,img.x2resol,img.a1data,img.a2data,1,delm,filename)


	print('\n\nEND OF CASE NUMBER:\t',filenum+1)
	print('\n\nEND OF CASE NAME:\t',filename)
	
