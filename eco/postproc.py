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
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator)
from mpfit import mpfit
import copy
import eco
#import time as t
import os
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#currentfolder = "E:/pramod_anchroedDrop/1_DC_damping_oscillations_analysis/twodrops_output"
currentfolder = "E:/pramod_anchroedDrop_Analysis/3_polarized_drops/1drop/DC/output"
print('\n folder of postprocessing output data is at : ',currentfolder)

###############1. writing parameters to file
def write_ecopar(par,imagedata,envdata,filename):
    print('\n writing experiment image parameters to file...\n')
    (fps,pxl_resol,radius1,radius2,density_drop,visc_drop,density_bulk,visc_bulk,permitivity_bulk,ift,V,s0,f_ac) =par
    radius = (radius1+radius2)/2
    r_diff = radius1-radius2
    mass1 =(4*np.pi/3)*(radius1**3) 
    mass2 =(4*np.pi/3)*(radius2**3) 
    m_diff = mass1-mass2
    #T1_eachcycle=envdata.Tosc1
    T1=envdata.Tosc1_avg
    #freq_osc1_eachcycle= [1/i for i in T1]
    freq_osc1= 1/T1  #freq from average time period
    T2=envdata.Tosc2_avg
    freq_osc2= 1/T2
    v1_initial =imagedata.v1data[0] #envdata.v1data[0]; from the 2nd data point of displacement
    v2_initial =imagedata.v2data[0] #envdata.v2data[0]
    v1_initial_model = eco.initialval(par)[1]
    v2_initial_model = eco.initialval(par)[3]
    x1max_by_radius=envdata.x1max[0]/radius1
    x2max_by_radius=envdata.x1max[0]/radius2
    x1max_by_s0 = envdata.x1max[0]/s0
    x2max_by_s0 = envdata.x2max[0]/s0

    y1=np.delete(np.abs(np.fft.fft(imagedata.x1data)),0)
    y2=np.delete(np.abs(np.fft.fft(imagedata.x2data)),0)
    y1=np.delete(y1,0)
    y2=np.delete(y2,0)
    y1=np.delete(y1,-1)
    y2=np.delete(y2,-1)
    y1=np.delete(y1,-1)
    y2=np.delete(y2,-1)
    y1=np.delete(y1,-1)
    y2=np.delete(y2,-1)
    y1=np.delete(y1,-1)
    y2=np.delete(y2,-1)
    freq = np.linspace(0, 1 / (imagedata.tdata[1] - imagedata.tdata[0]), len(imagedata.tdata))
    #freq2 = np.linspace(0, 1 / (imagedata.tdata[1] - imagedata.t2data[0]), len(imagedata.t2data))
    #freq1= np.delete(freq1,0)
    #freq2= np.delete(freq2,0)
    freq1_fft=freq[np.argmax(y1)+2]# +2 because we deleted first 2 elements of y1
    #freq2_fft= freq[y2.argmax(0)]
    freq2_fft= freq[np.argmax(y2)+2]

    head=['radius1[m]','radius2[m]','mass1[kg]','mass2[kg]','r_difference[m]','m_difference[kg]','time perdiod osc1[s]','time perdiod osc2[s]','freq1[Hz]','freq2[Hz]','freq1_fft','freq2_fft','initial velocity1[m/s]','initial velocity2[m/s]','x1max_by_radius','x2max_by_radius','x1max_by_s0','x2max_by_s0','s0_by_r_avg','ECOtheory_initial velocity1[m/s]','ECOtheory_initial velocity2[m/s]','f_ac']
    filespath = os.path.join(currentfolder, filename)
    if not os.path.exists(filespath):
      os.makedirs(filespath)
    with open(currentfolder+'/'+filename+'/'+'expdata_eco.csv', 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=head)
        writer.writeheader()
        #for i in range (0,4):
            #writer.writerow({'player': player_name[i], 'rating': fide_rating[i]})
        writer.writerow({head[0]: radius1,head[1]: radius2, head[2]: mass1,head[3]: mass2,head[4]: r_diff,head[5]:m_diff, head[6]:T1, head[7]:T2, head[8]:freq_osc1, head[9]:freq_osc2,head[10]:freq1_fft, head[11]:freq2_fft,head[12]:v1_initial, head[13]:v2_initial,head[14]:x1max_by_radius,head[15]:x2max_by_radius,head[16]:x1max_by_s0,head[17]:x2max_by_s0,head[18]:s0/radius,head[19]:v1_initial_model,head[20]:v2_initial_model,head[21]:f_ac})
    delm=2*m_diff/(mass1+mass2)
    return (delm)

def write_ehdpar(par,v_scale,filename):
    print('\n writing EHD parameters to file...\n')
    (fps,pxl_resol,radius1,radius2,density_drop,visc_drop,density_bulk,visc_bulk,permitivity_bulk,ift,V,s0,f_ac) =par
    radius = (radius1+radius2)/2
    E0 = V/s0
    Oh = visc_drop/np.sqrt(ift*2*radius*density_drop)    # Ohnesorge number based on diameter
    #Oh = visc_drop/np.sqrt(ift*2*radius1*density_drop)    # Ohnesorge number - single drop case based on diameter
    We = density_drop*v_scale**2*2*radius/ift      #based on diameter
    #We = density_drop*v_scale**2*2*radius1/ift  #radius2 for single drop case- based on diameter
    Ca_e =  permitivity_bulk*8.85e-12*E0**2*2*radius/ift  #based on diameter
    #Ca_e =  permitivity_bulk*8.85e-12*E0**2*2*radius1/ift #radius2 for single drop case -based on diameter
    Ca = visc_drop*v_scale/ift

    head=['nominal electric field strength[kV/cm]','ohnesorgeNum_(D)[1]','weber number_(D)','electric capillary_(D)','capillary number']
    filespath = os.path.join(currentfolder, filename)
    if not os.path.exists(filespath):
      os.makedirs(filespath)
    with open(currentfolder+'/'+filename+'/'+'expdata_ehd.csv', 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=head)
        writer.writeheader()
        #for i in range (0,4):
            #writer.writerow({'player': player_name[i], 'rating': fide_rating[i]})
        writer.writerow({head[0]: E0*1e-5,head[1]: Oh,head[2]: We,head[3]: Ca_e,head[4]: Ca})

def write_env(yaxis,imgenvfit,dynphase,filename):
    print('\n writing envelop fit parameters to file...\n')
    beta1=(imgenvfit.A1_1/(2*3.14/imgenvfit.omg_b1))/(imgenvfit.x1_ss/imgenvfit.tau_d1)
    beta2=(imgenvfit.A1_2/(2*3.14/imgenvfit.omg_b2))/(imgenvfit.x2_ss/imgenvfit.tau_d2)
    head=['dynamic phase','drop1 f_osc/f_beats','drop2 f_osc/f_beats','dynphase.xaxis','dynphase.yaxis','drop1 f_beats', 'drop2 f_beats','damping time scale of drop1 from env fit', 'damping time scale of drop2 from env fit','drop1 steady state displacement', 'drop2 steady state displacement','beta1(=(A1/Tb)/(A2/Td))','beta2','A1_1','A1_2','A3_1','A3_2','beats phase lead with osc of drop1','beats phase lead with osc of drop2']
    filespath = os.path.join(currentfolder, filename)
    if not os.path.exists(filespath):
      os.makedirs(filespath)
    with open(currentfolder+'/'+filename+'/'+'exp_envfit.csv', 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=head)
        writer.writeheader()
        writer.writerow({head[0]:dynphase.dynphase, head[1]:dynphase.ratio1, head[2]:dynphase.ratio2,head[3]:dynphase.xaxis,head[4]:yaxis ,head[5]:dynphase.omg_b1/(2*np.pi), head[6]: dynphase.omg_b2/(2*np.pi), head[7]: imgenvfit.tau_d1,head[8]:imgenvfit.tau_d2,head[9]:imgenvfit.x1_ss,head[10]:imgenvfit.x2_ss,head[11]:beta1,head[12]:beta2,head[13]:imgenvfit.A1_1,head[14]:imgenvfit.A1_2,head[15]:imgenvfit.A3_1,head[16]:imgenvfit.A3_2,head[17]:imgenvfit.phi_b1,head[18]:imgenvfit.phi_b2})

def write_fit(guessmodel, fitmodel,filename,key=2):
    print('\n writing ECO fit parameters to file...\n')
    #(del_m,k,del_k,zeta,del_zeta,f_e,s0)=coef
    #f1_nat = sqrt(k*(1+del_k))/(2*3.1417)
    #f2_nat = sqrt(k*(1-del_k))/(2*3.1417)
    guesscoef= guessmodel.thcoef
    if key ==1:
        fitcoef = fitmodel.thcoef
    else:
        fitcoef = fitmodel.optcoef
    head=['delm_guess','k_guess','delk_guess','zeta_guess','del zeta_guess','f_e_guess','resid_guess','delm_fit','k_fit','delk_fit','zeta_fit','del zeta_fit','f_e_fit','resid_fit','A1'] #,'f1_natural','f2_natural'
    filespath = os.path.join(currentfolder, filename)
    if not os.path.exists(filespath):
      os.makedirs(filespath)
    with open(currentfolder+'/'+filename+'/'+'exp_imagefit.csv', 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=head)
        writer.writeheader()
        writer.writerow({head[0]: guesscoef[0],head[1]: guesscoef[1], head[2]: guesscoef[2],head[3]: guesscoef[3],head[4]: guesscoef[4],head[5]:guesscoef[5],head[6]: "fitmodel.resid0",head[7]: fitcoef[0],head[8]: fitcoef[1], head[9]: fitcoef[2],head[10]: fitcoef[3],head[11]: fitcoef[4],head[12]:fitcoef[5],head[13]: "fitmodel.resid"})#,head[25]:f1_nat,head[26]:f2_nat

#############2. Dynamics plots
##A. Time series (of x,g,h,v,acceleration)
#def plttimeseries (t1data,x1data,t1model,x1model,t2data,x2data,t2model,x2model,filename,x1name,x2name): 
def plttimeseries_general (t1data,x1data,t2data,x2data,filename,title, subtitle1,subtitle2): 
    fig = plt.figure(figsize=(10, 5))
    gs = GridSpec(nrows=2, ncols=1)
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(t1data,x1data)
    ax0.xaxis.set_major_locator(MultipleLocator(t1data[-1]/10))
    ax0.xaxis.set_minor_locator(MultipleLocator(t1data[-1]/50))
    #x1ticks=np.arange(t1data[0],t1data[-1],0.005)
    #ax0.set_xticks(x1ticks)
    ax0.set_title(subtitle1) 
    #ax0.legend([label1])

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(t2data,x2data)
    ax1.xaxis.set_major_locator(MultipleLocator(t2data[-1]/10))
    ax1.xaxis.set_minor_locator(MultipleLocator(t2data[-1]/50))
    ax1.set_title(subtitle2)
    #ax1.legend([label2])

    plt.suptitle (title+filename)
    plt.draw()

    filespath = os.path.join(currentfolder, filename)
    if not os.path.exists(filespath):
      os.makedirs(filespath)
    plt.savefig(currentfolder+'/'+filename+'/'+title+filename+'.png')
    #plt.show()

def plttimeseries_general_two (t1data_1,x1data_1,t2data_1,x2data_1,t1data_2,x1data_2,t2data_2,x2data_2,filename,title, label1,label2): 
    fig = plt.figure(figsize=(10, 5))
    gs = GridSpec(nrows=2, ncols=1)
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.scatter(t1data_1,x1data_1)
    ax0.plot(t1data_2,x1data_2)
    ax0.xaxis.set_major_locator(MultipleLocator(t1data[-1]/10))
    ax0.xaxis.set_minor_locator(MultipleLocator(t1data[-1]/50))
    #x1ticks=np.arange(t1data[0],t1data[-1],0.005)
    #ax0.set_xticks(x1ticks)
    #ax0.set_title(label1) 
    ax0.legend([label1,label2])

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.scatter(t2data_1,x2data_1)
    ax1.plot(t2data_2,x2data_2)
    ax1.xaxis.set_major_locator(MultipleLocator(t2data[-1]/10))
    ax1.xaxis.set_minor_locator(MultipleLocator(t2data[-1]/50))
    #ax1.set_title(label2)
    ax1.legend([label1,label2])

    plt.suptitle (title+filename)
    plt.draw()

    filespath = os.path.join(currentfolder, filename)
    if not os.path.exists(filespath):
      os.makedirs(filespath)
    plt.savefig(currentfolder+'/'+filename+'/'+title+filename+'.png')
    #plt.show()

def plttimeseries (t1data,x1data,t2data,x2data,filename): 
    fig = plt.figure(figsize=(10, 5))
    gs = GridSpec(nrows=2, ncols=1)
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.scatter(t1data,x1data)
    ax0.xaxis.set_major_locator(MultipleLocator(t1data[-1]/10))
    ax0.xaxis.set_minor_locator(MultipleLocator(t1data[-1]/50))
    #x1ticks=np.arange(t1data[0],t1data[-1],0.005)
    #ax0.set_xticks(x1ticks)
    ax0.set_title('x1_raw data') 
    ax0.legend(['expt'])

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.scatter(t2data,x2data)
    ax1.xaxis.set_major_locator(MultipleLocator(t2data[-1]/10))
    ax1.xaxis.set_minor_locator(MultipleLocator(t2data[-1]/50))
    ax1.set_title('x1_raw data')
    ax1.legend(['expt'])

    plt.suptitle ('raw data time series plot for theory of: \n'+filename)
    plt.draw()

    filespath = os.path.join(currentfolder, filename)
    if not os.path.exists(filespath):
      os.makedirs(filespath)
    plt.savefig(currentfolder+'/'+filename+'/'+'tseries_raw_th_'+filename+'.png')
    #plt.show()

def plttimeseries_yscaled (t1data,x1data,t2data,x2data,filename,s0,rad1,rad2): 
    fig = plt.figure(figsize=(10, 5))
    gs = GridSpec(nrows=2, ncols=1)
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(t1data,x1data/s0,'-g')
    #ax0.plot(t1model,x1model,'-b')
    ax0.xaxis.set_major_locator(MultipleLocator(t1data[-1]/10))
    ax0.xaxis.set_minor_locator(MultipleLocator(t1data[-1]/50))
    ax0.set_title('x1') 
    ax0.legend(['expt'])

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(t2data,x2data/s0,'-g')
    #ax1.plot(t2model,x2model,'-b')
    ax1.xaxis.set_major_locator(MultipleLocator(t2data[-1]/10))
    ax1.xaxis.set_minor_locator(MultipleLocator(t2data[-1]/50))
    ax1.set_title('x2')
    ax1.legend(['expt'])

    plt.suptitle ('x1 & x2 scaled with s0 with \n scaled radii= '+str(rad1/s0)+' and '+str(rad2/s0)+' respectively')
    plt.draw()

    filespath = os.path.join(currentfolder, filename)
    if not os.path.exists(filespath):
      os.makedirs(filespath)
    plt.savefig(currentfolder+'/'+filename+'/'+'tseries_scaled_exp_'+filename+'.png')
    #plt.show()

def plttimeseries_g_h_scaled (tdata,x1data,x2data,filename,s0,rad1,rad2): 
    fig = plt.figure(figsize=(10, 5))
    gs = GridSpec(nrows=2, ncols=1)
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(tdata,(x1data+x2data)/s0,'-g')
    #ax0.plot(t1model,x1model,'-b')
    ax0.xaxis.set_major_locator(MultipleLocator(tdata[-1]/10))
    ax0.xaxis.set_minor_locator(MultipleLocator(tdata[-1]/50))
    ax0.set_title('x1') 
    ax0.legend(['expt'])

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(tdata,(x1data-x2data)/s0,'-g')
    #ax1.plot(t2model,x2model,'-b')
    ax1.xaxis.set_major_locator(MultipleLocator(tdata[-1]/10))
    ax1.xaxis.set_minor_locator(MultipleLocator(tdata[-1]/50))
    ax1.set_title('x2')
    ax1.legend(['expt'])

    plt.suptitle ('(x1+x2) & (x1-x2) scaled with s0 with \n scaled radii= '+str(rad1/s0)+' and '+str(rad2/s0)+' respectively')
    plt.draw()

    filespath = os.path.join(currentfolder, filename)
    if not os.path.exists(filespath):
      os.makedirs(filespath)
    plt.savefig(currentfolder+'/'+filename+'/'+'tseries_g_h_scaled_exp_'+filename+'.png')
    #plt.show()

def plttimeseries_envelops (imgdata,envdata,envfit,filename): 
    fig = plt.figure(figsize=(10, 5))
    gs = GridSpec(nrows=2, ncols=1)
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.scatter(imgdata.t1resol,imgdata.x1resol)
    ax0.scatter(envdata.t1max,envdata.x1max,label='_nolegend_')
    ax0.scatter(envdata.t1min,envdata.x1min,label='_nolegend_')
    ax0.plot(envfit.envfit1_max.t_env_fit,envfit.envfit1_max.x_env_fit)
    ax0.plot(envfit.envfit1_min.t_env_fit,envfit.envfit1_min.x_env_fit)
    #ax0.plot(t1model,x1model,'-b')
    ax0.xaxis.set_major_locator(MultipleLocator(imgdata.t1resol[-1]/10))
    ax0.xaxis.set_minor_locator(MultipleLocator(imgdata.t1resol[-1]/50))
    ax0.set_title('x1 data') 
    ax0.legend(['expt','max env fit','min env fit'])

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.scatter(imgdata.t2resol,imgdata.x2resol,label='Exp')
    ax1.scatter(envdata.t2max,envdata.x2max,label='_nolegend_')
    ax1.scatter(envdata.t2min,envdata.x2min,label='_nolegend_')
    ax1.plot(envfit.envfit2_max.t_env_fit,envfit.envfit2_max.x_env_fit,label='max env fit')
    ax1.plot(envfit.envfit2_min.t_env_fit,envfit.envfit2_min.x_env_fit,label='min env fit')
    #ax1.plot(t2model,x2model,'-b')
    ax0.xaxis.set_major_locator(MultipleLocator(imgdata.t2resol[-1]/10))
    ax0.xaxis.set_minor_locator(MultipleLocator(imgdata.t2resol[-1]/50))
    ax1.set_title('x2 data')
    #ax1.legend(['expt','max env fit','min env fit'])

    plt.suptitle ('Exp data with damping beating envelop: \n'+filename)
    plt.draw()

    filespath = os.path.join(currentfolder, filename)
    if not os.path.exists(filespath):
      os.makedirs(filespath)
    plt.savefig(currentfolder+'/'+filename+'/'+'tseries_exp_env_'+filename+'.png')
    #plt.show()

def plttimeseries_envelop_relax (imgdata,envdata,envfit,filename): 
    fig = plt.figure(figsize=(10, 5))
    gs = GridSpec(nrows=2, ncols=1)
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.scatter(imgdata.t1resol,[i * -1 for i in imgdata.x1resol])
    ax0.scatter(envdata.t1max,[i * -1 for i in envdata.x1max],label='_nolegend_')
    ax0.scatter(envdata.t1min,[i * -1 for i in envdata.x1min],label='_nolegend_')
    ax0.plot(envfit.envfit1_max.t_env_fit,[i * -1 for i in envfit.envfit1_max.x_env_fit])
    ax0.plot(envfit.envfit1_min.t_env_fit,[i * -1 for i in envfit.envfit1_min.x_env_fit])
    #ax0.plot(t1model,x1model,'-b')
    ax0.xaxis.set_major_locator(MultipleLocator(imgdata.t1resol[-1]/10))
    ax0.xaxis.set_minor_locator(MultipleLocator(imgdata.t1resol[-1]/50))
    ax0.set_title('x1 data') 
    ax0.legend(['expt','max env fit','min env fit'])

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.scatter(imgdata.t2resol,[i * -1 for i in imgdata.x2resol])
    ax1.scatter(envdata.t2max,[i * -1 for i in envdata.x2max],label='_nolegend_')
    ax1.scatter(envdata.t2min,[i * -1 for i in envdata.x2min],label='_nolegend_')
    ax1.plot(envfit.envfit2_max.t_env_fit, [i * -1 for i in envfit.envfit2_max.x_env_fit])
    ax1.plot(envfit.envfit2_min.t_env_fit, [i * -1 for i in envfit.envfit2_min.x_env_fit])
    #ax1.plot(t2model,x2model,'-b')
    ax0.xaxis.set_major_locator(MultipleLocator(imgdata.t2resol[-1]/10))
    ax0.xaxis.set_minor_locator(MultipleLocator(imgdata.t2resol[-1]/50))
    ax1.set_title('x2 data')
    ax1.legend(['expt','max env fit','min env fit'])

    plt.suptitle ('Exp data with damping beating envelop: \n'+filename)
    plt.draw()

    filespath = os.path.join(currentfolder, filename)
    if not os.path.exists(filespath):
      os.makedirs(filespath)
    plt.savefig(currentfolder+'/'+filename+'/'+'tseries_exp_env_'+filename+'.png')
    #plt.show()

def plttimeseries_fit (imgdata,envdata,envfit,modelfit,filename):
    fig = plt.figure(figsize=(10, 5))
    gs = GridSpec(nrows=2, ncols=1)
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.scatter(imgdata.t1resol,imgdata.x1resol,label='exp')
    print(len(envdata.t1max))
    print('\n\n',envdata.t1max,'\n\n')
    print('\n\n',envdata.x1max,'\n\n')
    print(len(envdata.x1max))
    ax0.scatter(envdata.t1max,envdata.x1max,label='_nolegend_')
    ax0.scatter(envdata.t1min,envdata.x1min,label='_nolegend_')
    ax0.plot(envfit.envfit1_max.t_env_fit,envfit.envfit1_max.x_env_fit)
    ax0.plot(envfit.envfit1_min.t_env_fit,envfit.envfit1_min.x_env_fit)
    ax0.plot(modelfit.t,modelfit.x1_manual,'-b')
    ax0.xaxis.set_major_locator(MultipleLocator(imgdata.t1resol[-1]/10))
    ax0.xaxis.set_minor_locator(MultipleLocator(imgdata.t1resol[-1]/50))
    ax0.set_title('x1 data') 
    ax0.legend(['expt','max envelop fit','min envelop fit','ECO fit'])

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.scatter(imgdata.t2resol,imgdata.x2resol)
    ax1.scatter(envdata.t2max,envdata.x2max,label='_nolegend_')
    ax1.scatter(envdata.t2min,envdata.x2min,label='_nolegend_')
    ax1.plot(envfit.envfit2_max.t_env_fit,envfit.envfit2_max.x_env_fit)
    ax1.plot(envfit.envfit2_min.t_env_fit,envfit.envfit2_min.x_env_fit)
    ax1.plot(modelfit.t,modelfit.x2_manual,'-b')
    ax1.xaxis.set_major_locator(MultipleLocator(imgdata.t2resol[-1]/10))
    ax1.xaxis.set_minor_locator(MultipleLocator(imgdata.t2resol[-1]/50))
    ax1.set_title('x2 data')
    ax1.legend(['expt','max envelop fit','min envelop fit','ECO fit'])

    plt.suptitle ('x1 and x2 exp & ECO model of : \n'+filename)
    plt.draw()

    filespath = os.path.join(currentfolder, filename)
    if not os.path.exists(filespath):
      os.makedirs(filespath)
    plt.savefig(currentfolder+'/'+filename+'/'+filename+'_tseries_exp_fit.png')
    #plt.show()

##B. phase plots
#def phaseplot1(x1data,x2data,x1model,x2model)
def phaseplot1(x1data,x2data,filename): #x2 vs x1
    fig = plt.figure(figsize=(5, 5))
    gs = GridSpec(nrows=1, ncols=1)
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(x1data,x2data,'-g')
    #ax0.plot(x1model,x2model,'-b')
    ax0.set_title('x2 vs x1') 
    ax0.legend(['expt'])
    plt.draw()

    filespath = os.path.join(currentfolder, filename)
    if not os.path.exists(filespath):
      os.makedirs(filespath)
    plt.savefig(currentfolder+'/'+filename+'/'+filename+'_phplot1.png')
    
    #plt.show()

def phaseplot2(x1data,x2data,v1data,v2data,filename): # velocity vs displacement
    x1_vdata = np.delete(x1data,0)
    x1_vdata = np.delete(x1_vdata,-1)
    x2_vdata = np.delete(x2data,0)
    x2_vdata = np.delete(x2_vdata,-1)

    fig = plt.figure(figsize=(10, 5))
    gs = GridSpec(nrows=2, ncols=1)
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(x1_vdata,v1data,'-g')
    #ax0.plot(v1model,x1model,'-b')
    ax0.set_title('v1 vs x1') 
    ax0.legend(['expt'])

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(x2_vdata,v2data,'-g')
    #ax1.plot(v2model,x2model,'-b')
    ax1.set_title('v2 vs x2')
    ax1.legend(['expt'])

    plt.suptitle ('velocity vs displacement')
    plt.draw()

    filespath = os.path.join(currentfolder, filename)
    if not os.path.exists(filespath):
      os.makedirs(filespath)
    plt.savefig(currentfolder+'/'+filename+'/'+filename+'_phplot2.png')
    #plt.show()

def phaseplot2_model(x1model,x2model,v1model,v2model,filename): # velocity vs displacement from ECO
    x1_vmodel = np.delete(x1model,0)
    x1_vmodel = np.delete(x1_vmodel,-1)
    x2_vmodel = np.delete(x2model,0)
    x2_vmodel = np.delete(x2_vmodel,-1)

    fig = plt.figure(figsize=(10, 5))
    gs = GridSpec(nrows=2, ncols=1)
    ax0 = fig.add_subplot(gs[0, 0])
    #ax0.plot(x1_vdata,v1data,'-g')
    ax0.plot(x1_vmodel,v1model,'-b')
    ax0.set_title('v1 vs x1') 
    ax0.legend(['model'])

    ax1 = fig.add_subplot(gs[1, 0])
    #ax1.plot(x2_vdata,v2data,'-g')
    ax1.plot(x2_vmodel,v2model,'-b')
    ax1.set_title('v2 vs x2')
    ax1.legend(['model'])

    plt.suptitle ('velocity vs displacement from ECO theory')
    plt.draw()

    filespath = os.path.join(currentfolder, filename)
    if not os.path.exists(filespath):
      os.makedirs(filespath)
    plt.savefig(currentfolder+'/'+filename+'/'+'phplot2_model_'+filename+'.png')
    #plt.show()

def fft(x1data,x2data,tdata,filename):
    numpoints=len(tdata)
    yf1=np.fft.fft(x1data)
    yf2=np.fft.fft(x2data)
    del_t = tdata[1] - tdata[0]  # sampling interval i.e., time step size
    #freq = np.linspace(0, 1 / del_t, numpoints)  # this gives too large steps
    freq = np.fft.fftfreq(numpoints, d=del_t)  #

    fig, ax = plt.subplots()
    #ax.plot(freq, np.abs(yf1),'-g',freq,np.abs(yf2),'-b') # 2drops case
    ax.plot(freq, np.abs(yf1),'-g',freq,np.abs(yf1),'-b') #1 drop case
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    plt.xlim(0,350) # this includes the something near 0 value, which sets the yaxis axis scale to be too long to visualize the plot
    plt.xlim(1/(numpoints+1),350)  #remember 350  is hardcoded upper limit of freq for fft.
    plt.ylim(1/(numpoints+1),max(max(np.abs(yf1[1:int(350*del_t*numpoints)])), max(np.abs(yf2[1:int(350*del_t*numpoints)]))))
    plt.legend(['drop1','drop2'])
    plt.grid(True)
    plt.suptitle ('FFT of drop pole displacement oscillations from experimental data')
    plt.draw()

    filespath = os.path.join(currentfolder, filename)
    if not os.path.exists(filespath):
      os.makedirs(filespath)
    y1=np.abs(yf1)
    y2=np.abs(yf2)
    y1=y1[:,np.newaxis]
    y2=y2[:,np.newaxis]
    freq=freq[:,np.newaxis]
    out=np.hstack((freq,y1,y2))
    np.savetxt(filespath+'/'+filename+'_fft.txt',out)
    
    plt.savefig(filespath+'/'+'fft.png')
    #plt.show()

##C. polar plots
def polarplot(theta1_data,theta2_data,x1_theta,x2_theta,filename):
    fig = plt.figure(figsize=(12,7))
    ax0 = fig.add_subplot(111, projection='polar')
    ax0.plot(theta1_data,x1_theta)
   # ax0.plot(theta2_data,x2_theta) #Disable for 1 drop case
    ax0.set_rticks([0.1*max(x1_theta), 0.25*max(x1_theta), 0.5*max(x1_theta), 1.0*max(x1_theta)])  # Less radial ticks
    ax0.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    ax0.grid(True)
    ax0.legend(['drop1','drop2'])

    plt.suptitle("Radial plot of displacement vs phase angle of individual oscillations", va='bottom')
    plt.draw()

    filespath = os.path.join(currentfolder, filename)
    if not os.path.exists(filespath):
      os.makedirs(filespath)
    plt.savefig(currentfolder+'/'+filename+'/'+filename+'_radial.png')
    #plt.show()

#############3. kinetics plots
def kineticsplot1(x1data,x2data,a1data,a2data,m,delm,filename): #force vs displacement
    x1_adata = np.delete(x1data,0)
    x2_adata = np.delete(x2data,0)
    x1_adata = np.delete(x1_adata,0)
    x2_adata = np.delete(x2_adata,0)
    x1_adata = np.delete(x1_adata,-1)
    x1_adata = np.delete(x1_adata,-1)
    x1_adata = np.delete(x1_adata,-1)
    x2_adata = np.delete(x2_adata,-1)
    x2_adata = np.delete(x2_adata,-1)
    x2_adata = np.delete(x2_adata,-1)

    fig = plt.figure(figsize=(10, 5))
    gs = GridSpec(nrows=2, ncols=1)
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(x1_adata,m*(1+delm)*np.asarray(a1data),'-g')
    #ax0.plot(v1model,x1model,'-b')
    ax0.set_title('net force vs x1') 
    ax0.legend(['expt'])

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(x2_adata,m*(1-delm)*np.asarray(a2data),'-g')
    #ax1.plot(v2model,x2model,'-b')
    ax1.set_title('net force vs x2')
    ax1.legend(['expt'])

    plt.suptitle ('net force vs displacement')
    plt.draw()

    filespath = os.path.join(currentfolder, filename)
    if not os.path.exists(filespath):
      os.makedirs(filespath)
    plt.savefig(currentfolder+'/'+filename+'/'+filename+'kineticsplot1.png')
    #plt.show()

def kineticsplot2(x1data,x2data,a1data,a2data,m,delm,filename): #energy vs displacement
    x1_adata = np.delete(x1data,0)
    x2_adata = np.delete(x2data,0)
    x1_adata = np.delete(x1_adata,0)
    x2_adata = np.delete(x2_adata,0)
    x1_adata = np.delete(x1_adata,-1)
    x1_adata = np.delete(x1_adata,-1)
    x1_adata = np.delete(x1_adata,-1)
    x2_adata = np.delete(x2_adata,-1)
    x2_adata = np.delete(x2_adata,-1)
    x2_adata = np.delete(x2_adata,-1)

    F1_integral =0    # reference energy
    F2_integral =0    # reference energy
    F1=[]
    F2=[]
    x1=[]
    for i in np.arange(0,len(a1data)):
        F1_integral=F1_integral+m*(1+delm)*a1data[i]*(x1data[i+1]-x1data[i-1])
        F1.append(F1_integral)
        #x1.append(x1_adata[i])
    for i in np.arange(0,len(a2data)):
        F2_integral=F2_integral+m*(1-delm)*a2data[i]*(x2data[i+1]-x2data[i-1])
        F2.append(F2_integral)
        #x2.append(x2_adata[i])
    fig = plt.figure(figsize=(10, 5))
    gs = GridSpec(nrows=2, ncols=1)
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(x1_adata,F1,'-g')
    #ax0.plot(v1model,x1model,'-b')
    ax0.set_title('energy vs x1') 
    ax0.legend(['expt'])

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(x2_adata,F2,'-g')
    #ax1.plot(v2model,x2model,'-b')
    ax1.set_title('energy vs x2')
    ax1.legend(['expt'])

    plt.suptitle ('energy landscape')
    plt.draw()

    filespath = os.path.join(currentfolder, filename)
    if not os.path.exists(filespath):
      os.makedirs(filespath)
    plt.savefig(currentfolder+'/'+filename+'/'+filename+'energy_plot.png')
    #plt.show()

#############4. Envelop and phase diagram
""" 
def envelopplot(img,envdata,filename):
    fig = plt.figure(figsize=(10, 5))

    gs = GridSpec(nrows=2, ncols=1)
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.scatter(img.tdata,img.x1data)
    ax0.scatter(envdata.t1max,envdata.x1max)
    ax0.scatter(envdata.t1min,envdata.x1min)
    ax0.plot(envdata.fit_env_1min.t_env_fit,envdata.fit_env_1min.x_env_fit,'-g')
    ax0.plot(envdata.fit_env_1max.t_env_fit,envdata.fit_env_1max.x_env_fit,'-g')
    ax0.xaxis.set_major_locator(MultipleLocator(0.005))
    ax0.xaxis.set_minor_locator(MultipleLocator(0.001))
    ax0.set_title('x1 vs t') 
    ax0.legend(['envelop fit'])

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.scatter(img.tdata,img.x2data)
    ax1.scatter(envdata.t2max,envdata.x2max)
    ax1.scatter(envdata.t2min,envdata.x2min)
    ax1.plot(envdata.fit_env_2min.t_env_fit,envdata.fit_env_2min.x_env_fit,'-g')
    ax1.plot(envdata.fit_env_2max.t_env_fit,envdata.fit_env_2max.x_env_fit,'-g')
    ax1.xaxis.set_major_locator(MultipleLocator(0.005))
    ax1.xaxis.set_minor_locator(MultipleLocator(0.001))
    ax1.set_title('x2 vs t') 
    ax1.legend(['envelop fit'])

    plt.suptitle('time series with envelop fitting to damping beating oscillations')
    plt.draw()

    filespath = os.path.join(currentfolder, filename)
    if not os.path.exists(filespath):
      os.makedirs(filespath)
    plt.savefig(currentfolder+'/'+filename+'/'+filename+'_tseries_envfit.png')
    plt.show()
""" 

def phasediagram(dynphase,xaxis,yaxis): # file loop above this call and plt.show() after the loop ends
    x_damping=[]
    x_beats=[]
    x_chaotic=[]
    y_damping=[]
    y_beats=[]
    y_chaotic=[]

    if dynphase== 'damping':
        x_damping.append(xaxis)
        y_damping.append(yaxis)
    elif dynphase== 'beats':
        x_damping.append(xaxis)
        y_damping.append(yaxis)
    elif dynphase == 'chaotic_oscillations':
        x_chaotic.append(xaxis)
        y_chaotic.append(yaxis)
    

    plt.plot(x_beats,y_beats,'ob',label='beats')
    plt.plot(x_damping,y_damping,'sg',label='damping')
    plt.plot(x_chaotic,y_chaotic,'vr',label='chaotic')
    plt.legend()
    plt.draw()              # add supercritically damped cases manually

    filespath = os.path.join(currentfolder, 'beats_phase_diagram')
    if not os.path.exists(filespath):
      os.makedirs(filespath)
    plt.savefig(filespath+'/'+'beats_phase_diagram.png')
    #plt.show()

#############(2. Plot trends from data) from excel origin 