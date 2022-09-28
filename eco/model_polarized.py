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
#import time as t
import os
import eco
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#print('current files location : ',os.getcwd())

###################### I. ECO model eqs, solvers & fitting algorithms #####################

def pol_thcoeff(p):
  [fps,pxl_resol,rad1,rad2,dens_d,visc_d,dens_b,visc_b,ift,perm_b,V,s0]=p
  r_0= (rad1+rad2)/2
  del_r = (rad1-rad2)/(rad1+rad2)
  del_m = (rad1**3-rad2**3)/(rad1**3+rad2**3)
  k=ift/(dens_d*r_0**3)
  del_k = del_m
  zeta = visc_d/np.sqrt(dens_d*ift*r_0)*1e4
  del_zeta = del_m
  #f_e =(3/4)*np.pi*perm_b*8.85*1e-12*V**2*r_0*(1-del_r**2)*(0.01/(1+r_0/s0))/s0**2*10
  f_ac='DC'

    #anchor_multiplication = np.pi*(5 + 3 Cos[2 \[Theta]o]) Sin[\[Theta]o]^2
    #del_m=0.025636032
    #s0=0.000353

    #(5.09705693e+06,6.04450821e-02,1.03401410e+02,3.93354126e-01,6.79374774e-06)
  k_factor=20.07
  delk_factor=1
  zeta_factor=40
  delzeta_factor=1
  fe_factor =2
  return(del_m,k_factor*k,delk_factor*del_k,zeta_factor*zeta,delzeta_factor*del_zeta,V,s0,f_ac)

def pol_initialval(p):
  [rad1,rad2,perm_b,V,s0]=[p[0],p[1],p[7],p[8],p[9]]

  c_Fe = (3*np.pi**2/8)*(0.01/(1+1/s0))*perm_b*8.85*1e-12/s0
  Fe1=c_Fe*V**2*rad1**3
  Fe2=c_Fe*V**2*rad2**3
  x_initial = [0,2*np.sqrt(Fe1),0,2*np.sqrt(Fe2)]
  return (x_initial)

def pol_nondimpar(p):
  [fps,pxl_resol,rad1,rad2,dens_d,visc_d,dens_b,visc_b,ift,perm_b,V,s0]=p
  #omega_n = dens_d
  Ca_e=perm_b*8.85*1e-12*V**2*(rad1+rad2)/(2*s0**2*ift)
  #Ca=visc_b*s0*omega_n
  Oh=visc_d/np.sqrt(dens_d*(rad1+rad2)*ift/2)
  #We=1
  #Bo=1
  return(Ca_e,Oh)

def pol_f_oper(x,t,*coef_tup): # eco operator returns the equations
  x1=x[0]
  y1=x[1]
  x2=x[2]
  y2=x[3]
  
  #print ('coef received by oper as :', type(coef_tup))
  coef = np.delete(coef_tup,-1).astype(np.float)
  #print('coeff as array: ',coef)
  tau_rise = 1e-5   #DC step rise time
  m=1
  del_m=coef[0]
  k=coef[1]
  del_k=coef[2]
  zeta=coef[3]
  del_zeta=coef[4]
  V=coef[5]
  s0=coef[6]

  r_0 = 0.000396
  del_r = 3e-6

  perm_b=1
  tau_pol =100
  alpha_pol= 1.2

  fe1 = (3/4)*np.pi*perm_b*8.85*1e-12*V**2*r_0*(1-del_r**2)*(0.01/(1+r_0/s0))/s0**2*10
  
  m1=(1.-del_m) # m=1
  m2=(1.+del_m) # m=1
  
  k1=k*(1.-del_k)
  k2=k*(1.+del_k)

  zeta1=zeta*(1.-del_zeta)
  zeta2=zeta*(1.+del_zeta)

  print(t,np.sinh(tau_pol*(V*1e-9)/(s0 - x1 - x2))/(tau_pol*(V*1e-9)/(s0 - x1 - x2)), (s0-x1-x2),tau_pol*V*1e-9/(s0 - x1 - x2))

  if coef_tup[7]=='DC':
    fe2 = (2*alpha_pol/tau_pol)*np.log(np.sinh(tau_pol*V/(s0 - x1 - x2))/(tau_pol*V/(s0 - x1 - x2)))
    f_op = [y1,
         (-k1 * x1/m1 - zeta1 * y1/m1 + (fe1/m1) / (s0 - x1 - x2)**2.) + fe2/m1,
         y2,
         (-k2 * x2/m2 - zeta2 * y2/m2+ (fe1/m2)/(s0 - x1 - x2)**2.) + fe2/m1]
  else:
    fe2_ac = (2*alpha_pol/tau_pol)*np.log(np.sinh(tau_pol*V*np.sin(2*np.pi*eval(f_ac)*t)/(s0 - x1 - x2))/(tau_pol*V*np.sin(2*np.pi*eval(f_ac)*t)/(s0 - x1 - x2)))
    f_ac = coef_tup[7]
    f_op = [y1,
         (-k1 * x1/m1 - zeta1 * y1/m1 + (fe1*np.sin(2*np.pi*eval(f_ac)*t)/m1) / (s0 - x1 - x2)**2.)+ fe2_ac/m1,
         y2,
         (-k2 * x2/m2 - zeta2 * y2/m2+ (fe1*np.sin(2*np.pi*eval(f_ac)*t)/m2)/(s0 - x1 - x2)**2.) + fe2_ac/m1]
  return f_op #need to keep coeff in () ?

"""
def pol_f_oper_AC(x,t,*coef_tup): # eco operator returns the equations
  x1=x[0]
  y1=x[1]
  x2=x[2]
  y2=x[3]
  
  #print ('coef received by oper as :', type(coef_tup))
  coef = np.asarray(coef_tup)
  #print('coeff as array: ',coef)
  tau_rise = 1e-5   #DC step rise time
  m=1
  del_m=coef[0]
  k=coef[1]
  del_k=coef[2]
  zeta=coef[3]
  del_zeta=coef[4]
  fe=coef[5]
  s0=coef[6]
  
  m1=(1.-del_m) # m=1
  m2=(1.+del_m) # m=1
  
  k1=k*(1.-del_k)
  k2=k*(1.+del_k)

  zeta1=zeta*(1.-del_zeta)
  zeta2=zeta*(1.+del_zeta)

  f_op = [y1,
         (-k1 * x1/m1 - zeta1 * y1/m1 + (fe*sin(2*np.pi*f_ac*t)/m1) / (s0 - x1 - x2)**2.),
         y2,
         (-k2 * x2/m2 - zeta2 * y2/m2+ (fe*sin(2*np.pi*f_ac*t)/m2)/(s0 - x1 - x2)**2.)]
  return f_op #need to keep coeff in () ?
"""

def pol_sol(t,coef,x0,tol): #numerical solution of the model
  ys = spi.odeint(pol_f_oper, x0, t, args=tuple(coef), atol=tol[0], rtol=tol[1])
  #print (ys[:,2])
  return (ys[:,0],ys[:,1],ys[:,2],ys[:,3])
  
def pol_fitsol(t,thcoef,fitcoef,x0,tol,numdrops):
  coef=[None]*8
  coef[0] = thcoef[0]
  coef[1]=thcoef[1]*fitcoef[0]
  coef[2]=thcoef[2]*fitcoef[1]
  coef[3]=thcoef[3]*fitcoef[2]
  coef[4]=thcoef[4]*fitcoef[3]
  coef[5]=thcoef[5]*fitcoef[4]
  coef[6]=thcoef[6]
  coef[7]=thcoef[7]
  #print('\n\n coef is: \n\n')
  #print(thcoef[1])
  coef=tuple(coef)

  ys = spi.odeint(f_oper, x0, t, args=(coef), atol=tol[0], rtol=tol[1],tfirst=False)

  if numdrops==2:
    x_1dmodel=np.asarray(list(np.array([ys[:,0],ys[:,2]]).flat))
  elif numdrops==1:  #only drop 2
    x_1dmodel=ys[:,2]

  return (x_1dmodel) # solve the model to return both oscilaltors data in a single array

def pol_residual(x_model,import_data):
  print('\n \n \n x1ddata: ', import_data[:,1])
  print('\n \n \n x1ddata shape: ', np.shape(import_data[:,1]))
  x_1dmodel=np.asarray(list(np.array([x_model[0],x_model[2]]).flat))
  x_1ddata=np.asarray(list(np.array([import_data[:,1],import_data[:,2]]).flat))
  #print('\n \n \n x1dmodel: ', x_1dmodel)
  #print('\n \n \n x1ddata: ', x_1ddata)
  resid = ((x_1dmodel-x_1ddata)**2.).sum()
  return (resid) #residual of a model coeff with the data

def pol_fitalgo(imgdata,thcoef,x0,tol,numdrops): #Global optimization to give fitted coefficients of eco model; del_m,s0 are fixed
  tdata=imgdata.tdata
  x1data=imgdata.x1data
  x2data=imgdata.x2data
  x_1ddata=np.asarray(list(np.array([x1data,x2data]).flat))
  stoptime=imgdata.tend
  numpoints=imgdata.numpoints
  fitcoef = [1,1,1,1,1] # initializing the array of fitting parameters:[c_k,c_delk,c_zeta,c_delzeta, c_e]
  #coef_var =(coef[1],coef[2],coef[3],coef[4],coef[5])
  #coef_fix =(coef[0],coef[6])

  if numdrops==2:
    res0 =(fitsol(tdata,thcoef,fitcoef,x0,tol,numdrops)- x_1ddata)**2.
    resid0 = sum(res0)
  elif numdrops==1: #only drop 2
    res0 =(fitsol(tdata,thcoef,fitcoef,x0,tol,numdrops)- x2data)**2.
    resid0 = sum(res0)

  #bound=([0.7*coef[1], 1.3*coef[1]],[0.7*coef[2], 1.3*coef[2]],[0.7*coef[3], 1.3*coef[3]],[0.7*coef[4], 1.3*coef[4]],[0.7*coef[5], 1.3*coef[5]])
  bound=([0.5, 3],[0.5, 5],[0.5, 2],[0.5, 5],[0.5, 3])
  
  def resid (fitcoef): # residual of fit to be used during the optimization
    #coef=(coef_fix[0],coef_var[0],coef_var[1],coef_var[2],coef_var[3],coef_var[4],coef_fix[1])
    #fitcoef=(fitcoef[0],fitcoef[1],fitcoef[2]) #if required to change the object type of array
    if numdrops==2:
      res =(fitsol(tdata,thcoef,fitcoef,x0,tol,numdrops)- x_1ddata)**2.
      resid = sum(res)
    elif numdrops==1:#only drop 2
      res =(fitsol(tdata,thcoef,fitcoef,x0,tol,numdrops)- x2data)**2.
      resid = sum(res0)
    return resid

  result = optimize.dual_annealing(resid,bounds=bound,maxiter=100)

  opt_fitcoef = result.x # optimal fit parameters
  #coef_opt = (coef_fix[0],coef_opt_var[0],coef_opt_var[1],coef_opt_var[2],coef_opt_var[3],coef_opt_var[4],coef_fix[1])
  x_1dsim = fitsol(tdata, thcoef,opt_fitcoef,x0,tol,numdrops)
  if numdrops ==2:
    residu = ((x_1dsim-x_1ddata)**2.).sum()
  elif numdrops ==1: #only drop 2
    residu = ((x_1dsim-x2data)**2.).sum() 
  print ('\n Optimal values of fitting coefficients of ECO model for the experimental data: ',opt_fitcoef)
  print ('\n least square residual of fit: ',residu)
  #print ('correlation of fit with experimental data: ',correl)
  return (opt_fitcoef,residu,resid0)

def pol_fit(optiter,imgdata,thcoef,x0,tol,numdrops):
  print('\n optimizing the ECO fit...\n')
  resid=[]
  fit_iter=[]
  for i in range(1,optiter+1):
      print ('\n \n optimization iteration :', i)
      fit_iter.append(fitalgo(imgdata,thcoef,x0,tol,numdrops)) #Numerically optimized coefficients
      #print ('\n optimiz iter residual :', fit_iter[i-1][1])
      resid.append(fit_iter[i-1][1]) # optimization iterations
  best_iter= resid.index(min(resid))
  opt_fitcoef=fit_iter[best_iter][0]
  resid_fit=fit_iter[best_iter][1]
  resid0_fit=fit_iter[best_iter][2]
  print ('\nbest iteration of optim is :', best_iter+1)
  print ('\nbest iteration optim fit coeff are :', opt_fitcoef)
  print ('\nbest iteration residual :', resid_fit)
  return (opt_fitcoef,resid_fit,resid0_fit)

class pol_thmodel(object):
      """" model data object is generated with the input of fluid, operating and geometric parameters of each case. 
      It consists of t,x1,x2 data from the model with the theoretical coeff"""
      def __init__(self, par,t,x0):
        self.tol =[1.0e-8, 1.0e-6] # default values of [abserr, relerr]
        self.par = par
        self.t=t
        self.x0=x0
        (self.Ca_e,self.Oh)=nondimpar(self.par)
        self.thcoef=thcoeff(self.par)
        self.del_m=self.thcoef[0]
        self.k=self.thcoef[1]
        self.del_k=self.thcoef[2]
        self.zeta=self.thcoef[3]
        self.del_zeta=self.thcoef[4]
        self.f_e=self.thcoef[5]
        self.s0=self.thcoef[6]
        self.f_ac = self.thcoef[7]
        self.x_th= sol(self.t,self.thcoef,self.x0,self.tol) #Theoretical solution
        self.x1_th =self.x_th[0]
        self.x2_th =self.x_th[2]

class pol_manual_model(object):
      """" model with completely manually selected coefficient values"""
      def __init__(self,coef,t,x0):
        self.tol =[1.0e-8, 1.0e-6] # default values of [abserr, relerr]
        self.t=np.linspace(t[0],t[-1],len(t)*100)
        self.x0=x0
        self.coef=coef
        (self.del_m,self.k,self.del_k,self.zeta,self.del_zeta,self.f_e,self.s0,f_ac)=self.coef
        self.x_manual= pol_sol(self.t,self.coef,self.x0,self.tol) #manual solution
        self.x1_manual =self.x_manual[0]
        self.x2_manual =self.x_manual[2]
        #self.resol=(self.t,self.x1_manual,self.t,self.x2_manual)
#        (self.v1model,self.v2model,self.a1model,self.a2model,self.theta1_model,self.theta2_model,self.x1_theta_model,self.x2_theta_model)=eco.data_derivative_par((self.t,self.x1_manual,self.t,self.x2_manual))
        # velocity, aceleration from formula instead !!? as this is formula based fitting

########################## IIA. (temporary) From Devosmita's fitting work ##############################
def pol_dev_emperical(p):
  [fps,pxl_resol,rad1,rad2,dens_d,visc_d,dens_b,visc_b,ift,perm_b,V,s0]=p
  k_factor=2.94761e-4
  delk_factor=1
  zeta_factor=1
  delzeta_factor=1
  fe_factor =9.44-14

  r_0= (rad1+rad2)/2
  del_m = (rad1**3-rad2**3)/(rad1**3+rad2**3)
  k=k_factor*r_0**3
  del_k = del_m
  zeta = visc_d/(dens_d*r_0**2)
  del_zeta = del_m
  f_e =fe_factor*r_0*(V/s0)**2
  f_ac = 'DC'

  return(del_m,k_factor*k,delk_factor*del_k,zeta_factor*zeta,delzeta_factor*del_zeta,fe_factor*f_e,s0,f_ac)

class pol_dev_empmodel(object):
      """" model data object is generated with the input of fluid, operating and geometric parameters of each case. 
      It consists of t,x1,x2 data from the model with the theoretical coeff"""
      def __init__(self, par,t,x0):
        self.tol =[1.0e-8, 1.0e-6] # default values of [abserr, relerr]
        self.par = par
        self.t=t
        self.x0=x0
        (self.Ca_e,self.Oh)=nondimpar(self.par)
        self.thcoef=dev_emperical(self.par)
        self.del_m=self.thcoef[0]
        self.k=self.thcoef[1]
        self.del_k=self.thcoef[2]
        self.zeta=self.thcoef[3]
        self.del_zeta=self.thcoef[4]
        self.f_e=self.thcoef[5]
        self.s0=self.thcoef[6]
        self.f_ac = self.thcoef[7]
        self.x_th= sol(self.t,self.thcoef,self.x0,self.tol) #Theoretical solution
        self.x1_th =self.x_th[0]
        self.x2_th =self.x_th[2]

####################### II. ECO fitting on envelop of oscillations data #########################
def pol_fit_env_min_iter(optiter,envdata_t,envdata_x):
  resid=[]
  fit_iter=[]
  for i in range(1,optiter+1):
      #print ('\n \n env min fit iter no :', i)
      fit_iter.append(fitenv_min(envdata_t,envdata_x)) #Numerically optimized coefficients
      #print ('\n optimiz iter residual :', fit_iter[i-1][1])
      resid.append(fit_iter[i-1][1]) # optimization iterations
  best_iter= resid.index(min(resid))
  opt_fitcoef=fit_iter[best_iter][0]
  x_ss=fit_iter[best_iter][0][1]
  resid_fit=fit_iter[best_iter][1]
  t_env_fit=fit_iter[best_iter][2]
  x_env_fit=fit_iter[best_iter][3]
  print ('\nbest iteration of min envelop optim is :', best_iter+1)
  print ('\nbest iteration optim min envelop fit coeff are :', opt_fitcoef)
  #print ('\nbest iteration steady state displacement :', x_ss)
  print ('\nbest iteration residual :', resid_fit)
  return (opt_fitcoef,resid_fit,t_env_fit,x_env_fit)

def pol_fit_env_max_iter(optiter,envdata_t,envdata_x):
  resid=[]
  fit_iter=[]
  for i in range(1,optiter+1):
      #print ('\n \n  env max fit iter no :', i)
      fit_iter.append(fitenv_max(envdata_t,envdata_x)) #Numerically optimized coefficients
      #print ('\n optimiz iter residual :', fit_iter[i-1][1])
      resid.append(fit_iter[i-1][1]) # optimization iterations
  best_iter= resid.index(min(resid))
  opt_fitcoef=fit_iter[best_iter][0]
  resid_fit=fit_iter[best_iter][1]
  t_env_fit=fit_iter[best_iter][2]
  x_env_fit=fit_iter[best_iter][3]
  print ('\nbest iteration of max fit optim is :', best_iter+1)
  print ('\nbest iteration max envelop optim fit coeff are :', opt_fitcoef)
  print ('\nbest iteration residual :', resid_fit)
  return (opt_fitcoef,resid_fit,t_env_fit,x_env_fit)

def pol_fitenv_min(envdata_t,envdata_x):
  #t_data1 =(envdata_t[0],envdata_t[len(envdata_t)-1],(envdata_t[1]-envdata_t[0])/4)
  #t_model = np.asarray(t_data1)
  t_data = np.asarray(envdata_t)
  envpar = [0,0,0,0,0] #A1,A2,tau_d,omg_b,phi
  #f_ac = 
  #bound=[[0.5e-5,2e-5],[5e-5,10e-5],[0.1,0.2],[2*np.pi/0.09,2*np.pi/0.075],[np.pi/3,3*np.pi/5]]
  bound=[[1e-6,9e-5],[1e-5,15e-5],[0.001,0.5],[2*np.pi/0.5,2*np.pi/0.01],[-np.pi,np.pi]]
  def residenvfit(envpar):
    A1=envpar[0]
    A2=envpar[1]
    tau_d=envpar[2]
    omg_b=envpar[3]
    phi=envpar[4]
    #(A1,A2,tau_d,omg_b,phi)=np.asarray(envpar)
    
    #if f_ac=='DC':
    x_env_model = np.asarray(np.exp(-1.0*t_data/tau_d)*(A1*np.sin(omg_b*t_data+phi)-A2)+A2)
    #else:

    #print(type(envdata_x))
    #print(type(x_env_model))
    return np.sqrt(sum((envdata_x-x_env_model)**2))
  
  result = optimize.dual_annealing(residenvfit,bounds=bound,maxiter=1000)
  opt_fitcoef = result.x
  x_env_fit = np.asarray(np.exp(-t_data/opt_fitcoef[2])*(opt_fitcoef[0]*np.sin(opt_fitcoef[3]*t_data+opt_fitcoef[4])-opt_fitcoef[1])+opt_fitcoef[1])
  residu = np.sqrt(((envdata_x-x_env_fit)**2).sum())

  t_model=np.linspace(t_data[0],t_data[-1],20*len(t_data))
  x_env_fit = np.asarray(np.exp(-t_model/opt_fitcoef[2])*(opt_fitcoef[0]*np.sin(opt_fitcoef[3]*t_model+opt_fitcoef[4])-opt_fitcoef[1])+opt_fitcoef[1])
  #print ('\n Fitting coefficients of envelop: ',x_env_fit)
  #print ('\n least square residual of fit: ',residu)
  #print ('correlation of fit with experimental data: ',correl)
  return (opt_fitcoef,residu,t_model,x_env_fit)

def pol_fitenv_max(envdata_t,envdata_x):
  #t_data1 =(envdata_t[0],envdata_t[len(envdata_t)-1],(envdata_t[1]-envdata_t[0])/4)
  #t_data = np.asarray(t_data1)
  t_data = np.asarray(envdata_t)
  envpar = [0,0,0,0,0,0]
  #bound=[[0.5e-5,2e-5],[5e-5,10e-5],[0.1,0.2],[2*np.pi/0.09,2*np.pi/0.075],[np.pi/3,3*np.pi/5]]
  bound=[[1e-6,9e-5],[1e-5,15e-5],[0.001,0.5],[2*np.pi/0.5,2*np.pi/0.01],[-np.pi,np.pi],[1e-6,15e-4]]
  def residenvfit(envpar):
    A1=envpar[0]
    A2=envpar[1]
    tau_d=envpar[2]
    omg_b=envpar[3]
    phi=envpar[4]
    A3=envpar[5]
    #(A1,A2,tau_d,omg_b,phi)=np.asarray(envpar)
    x_env_model = np.asarray(np.exp(-1.0*t_data/tau_d)*(A1*np.sin(omg_b*t_data+phi)+A3)+A2)
    return np.sqrt(sum((envdata_x-x_env_model)**2))
  
  result = optimize.dual_annealing(residenvfit,bounds=bound,maxiter=1000)
  opt_fitcoef = result.x
  x_env_fit = np.asarray(np.exp(-t_data/opt_fitcoef[2])*(opt_fitcoef[0]*np.sin(opt_fitcoef[3]*t_data+opt_fitcoef[4])+opt_fitcoef[5])+opt_fitcoef[1])
  residu = np.sqrt(((envdata_x-x_env_fit)**2).sum())

  t_model=np.linspace(t_data[0],t_data[-1],20*len(t_data))
  x_env_fit = np.asarray(np.exp(-t_model/opt_fitcoef[2])*(opt_fitcoef[0]*np.sin(opt_fitcoef[3]*t_model+opt_fitcoef[4])+opt_fitcoef[5])+opt_fitcoef[1])
  #print ('\n Fitting coefficients of envelop: ',x_env_fit)
  #print ('\n least square residual of fit: ',residu)
  #print ('correlation of fit with experimental data: ',correl)
  return (opt_fitcoef,residu,t_model,x_env_fit)

class pol_envelopfit_min(object):
  def __init__(self,envdata_t,envdata_x):
    self.optiter =8     # default number of iterations of global optim
    self.envdata_t = envdata_t
    self.envdata_x = envdata_x
    self.fit = fit_env_min_iter(self.optiter,self.envdata_t,self.envdata_x)
    self.opt_fitcoef= self.fit[0]
    (self.A1,self.A2,self.tau_d,self.omg_b,self.phi) = self.fit[0]
    #print('\n\n\n\n',self.tau_d,'\n\n\n\n')
    self.residual=self.fit[1]
    self.t_env_fit =self.fit[2]
    self.x_env_fit =self.fit[3]
    self.x_min_ss=self.A2
    #self.x_max_ss=self.A2 # for drop relaxation case

class pol_envelopfit_max(object):
  def __init__(self,envdata_t,envdata_x):
    self.optiter =8
    self.envdata_t = envdata_t
    self.envdata_x = envdata_x
    self.fit = fit_env_max_iter(self.optiter,envdata_t,envdata_x)
    self.opt_fitcoef= self.fit[0]
    (self.A1,self.A2,self.tau_d,self.omg_b,self.phi,self.A3) = self.fit[0]
    self.residual=self.fit[1]
    self.t_env_fit =self.fit[2]
    self.x_env_fit =self.fit[3]
    self.x_max_ss = self.A2
    #self.x_min_ss = self.A2 # for drop relaxation case

class pol_envfit(object):
  """docstring for envfit"""
  def __init__(self, envdata):
    """ 
    if envdata.t1min[0]<envdata.t1max[0]:
      t1min_envfit=np.delete(envdata.t1min,0)
      x1min_envfit=np.delete(envdata.x1min,0)
      t1max_envfit=envdata.t1max
      x1max_envfit=envdata.x1max
    elif envdata.t1min[0]>envdata.t1max[0]:
      t1max_envfit=np.delete(envdata.t1max,0)
      x1max_envfit=np.delete(envdata.x1max,0)
      t1min_envfit=envdata.t1min
      x1min_envfit=envdata.x1min

    if envdata.t2min[0]<envdata.t2max[0]:
      t2min_envfit=np.delete(envdata.t2min,0)
      x2min_envfit=np.delete(envdata.x2min,0)
      t2max_envfit=envdata.t2max
      x2max_envfit=envdata.x2max
    elif envdata.t2min[0]>envdata.t2max[0]:
      t2max_envfit=np.delete(envdata.t2max,0)
      x2max_envfit=np.delete(envdata.x2max,0)
      t2min_envfit=envdata.t2min
      x2min_envfit=envdata.x2min

    print('\nfitting x1 max envelop in to ECO...\n')
    self.envfit1_max=eco.envelopfit_max(t1max_envfit,x1max_envfit)
    print('\nfitting x1 min envelop in to ECO...\n')
    self.envfit1_min=eco.envelopfit_min(t1min_envfit,x1min_envfit)
    print('\nfitting x2 max envelop in to ECO...\n')
    self.envfit2_max=eco.envelopfit_max(t2max_envfit,x2max_envfit)
    print('\nfitting x2 min envelop in to ECO...\n')
    self.envfit2_min=eco.envelopfit_min(t2min_envfit,x2min_envfit)
    """ 
    #if numdrops==2:
    print('\nfitting x1 max envelop in to ECO...\n')
    self.envfit1_max=eco.envelopfit_max(envdata.t1max,envdata.x1max)
    print('\nfitting x1 min envelop in to ECO...\n')
    self.envfit1_min=eco.envelopfit_min(envdata.t1min,envdata.x1min)
    print('\nfitting x2 max envelop in to ECO...\n')
    self.envfit2_max=eco.envelopfit_max(envdata.t2max,envdata.x2max)
    print('\nfitting x2 min envelop in to ECO...\n')
    self.envfit2_min=eco.envelopfit_min(envdata.t2min,envdata.x2min)
    #elif numdrops==1:
    print('\nfitting x2 max envelop in to ECO...\n')
    self.envfit2_max=eco.envelopfit_max(envdata.t2max,envdata.x2max)
    print('\nfitting x2 min envelop in to ECO...\n')
    self.envfit2_min=eco.envelopfit_min(envdata.t2min,envdata.x2min)
    #else:
      #print('Error: Number of drops for envelop fitting can be only either one or two.')

    """
    self.envfit1_max=eco.envelopfit_min(envdata.t1max,envdata.x1max) # for drop relaxation case
    self.envfit1_min=eco.envelopfit_max(envdata.t1min,envdata.x1min)  # for drop relaxation case
    self.envfit2_max=eco.envelopfit_min(envdata.t2max,envdata.x2max)  # for drop relaxation case
    self.envfit2_min=eco.envelopfit_max(envdata.t2min,envdata.x2min)  # for drop relaxation case
    """
    self.A1_1=(self.envfit1_max.A1+self.envfit1_min.A1)/2
    self.A1_2=(self.envfit2_max.A1+self.envfit2_min.A1)/2
    self.A3_1=self.envfit1_max.A3
    self.A3_2=self.envfit2_max.A3
    self.omg_b1=(self.envfit1_max.omg_b+self.envfit1_min.omg_b)/2
    self.omg_b2=(self.envfit2_max.omg_b+self.envfit2_min.omg_b)/2 #omg_b1=(envelopfit_min(envdata.t1max,envdata.x1max).omg_b+envelopfit_min(envdata.t1min,envdata.x1min).omg_b)/2
    self.tau_d1=(self.envfit1_max.tau_d+self.envfit1_min.tau_d)/2
    self.tau_d2=(self.envfit2_max.tau_d+self.envfit2_min.tau_d)/2
    self.x1_ss=(self.envfit1_max.x_max_ss+self.envfit1_min.x_min_ss)/2
    self.x2_ss=(self.envfit2_max.x_max_ss+self.envfit2_min.x_min_ss)/2
    self.phi_b1=(self.envfit1_max.phi+self.envfit1_min.phi)/2
    self.phi_b2=(self.envfit2_max.phi+self.envfit2_min.phi)/2

    Tosc1_average=envdata.Tosc1_avg
    print('Tosc1_average =',Tosc1_average)
    Tosc2_average=envdata.Tosc2_avg
    print('Tosc2_average =', Tosc2_average)
    self.omg_osc1= 2*np.pi/(Tosc1_average)
    self.omg_osc2= 2*np.pi/(Tosc2_average)

####################### III. ECO fitting on complete oscillations data #########################

#def envfit_to_guesscoeff(par,envelopdata,envfit_x1_min,envfit_x2_min,envfit_x1_max,envfit_x2_max):
def pol_envfit_to_guesscoeff(par,envfit,numdrops):
  print('\nInitial guess of ECO parameters from the envelop fit on to images...\n')
  #Tosc1_avg=envelopdata.Tosc1_avg
  #print('Tosc1_avg= ', Tosc1_avg)
  #Tosc2_avg=envelopdata.Tosc2_avg
  #print('Tosc1_avg= ', Tosc1_avg)
  
  s0=par[11]
  #tau_d1_min= envfit_x1_min.tau_d
  #tau_d1_max= envfit_x1_max.tau_d
  #tau_d1 = (tau_d1_min+tau_d1_max)/2
  print('tau_d1= ', envfit.tau_d1)
  #tau_d2_min= envfit_x2_min.tau_d
  #tau_d2_max= envfit_x2_max.tau_d
  #tau_d2 = (tau_d2_min+tau_d2_max)/2
  print('tau_d2= ', envfit.tau_d2)
  print('envfit.x1_ss= ', envfit.x1_ss)
  print('envfit.x2_ss= ', envfit.x2_ss)
  #omg_osc1= 2*np.pi/(envfit.Tosc1_avg)
  #omg_osc2= 2*np.pi/(envfit.Tosc2_avg)
  #x1_ss = (envfit_x1_min.A2+envfit_x1_max.A2)/2
  #x2_ss = (envfit_x2_min.A2+envfit_x2_max.A2)/2

  if numdrops ==2:
    delm=(par[3]**3-par[2]**3)/(par[3]**3+par[2]**3)
    zeta = ((1-delm)/envfit.tau_d1)+((1+delm)/envfit.tau_d2) #2*(1-delm)/(tau_d1*(1-del_zeta)) # (1/tau_d1 - 1/tau_d2)*(1-delm) 
    del_zeta = (1/zeta)*(((1+delm)/envfit.tau_d2)-((1-delm)/envfit.tau_d1)) #(tau_d1-tau_d2+delm*(tau_d1+tau_d2))/(tau_d1+tau_d2+delm*(tau_d1-tau_d2)) #(1/zeta)*((1/tau_d2 - 1/tau_d1)+delm*(1/tau_d2 + 1/tau_d1)) 
    k =0.5*( (1-delm)*envfit.omg_osc1**2 + (1+delm)*envfit.omg_osc2**2 + (zeta*(1-del_zeta))**2/(4*(1-delm))+ (zeta*(1+del_zeta))**2/(4*(1+delm)) )#omg_osc1**2+(zeta*(1+del_zeta)/(2*(1+delm))) #0.5*((1-delm)*omg_osc1**2 + (1+delm)*omg_osc2**2 + zeta)
    delk =(0.5/k)*((1+delm)*envfit.omg_osc2**2-(1-delm)*envfit.omg_osc1**2+(zeta*(1+del_zeta))**2/(4*(1+delm)) + (zeta*(1-del_zeta))**2/(4*(1-delm)) )  #(0.5/k)*((omg_osc2**2-omg_osc1**2) + delm*(omg_osc2**2+omg_osc1**2) + zeta*del_zeta)
    f_e = (k*(envfit.x1_ss+envfit.x2_ss)+k*delk*(-envfit.x1_ss+envfit.x2_ss))*0.5*(s0-envfit.x1_ss-envfit.x2_ss)**2
    f_ac = par[12]

  elif numdrops ==1:
    delm=0.0
    zeta = (2/envfit.tau_d2 )
    del_zeta = 0.0
    k = (envfit.omg_osc2**2 +(zeta**2)/4)
    delk = 0.0
    #f_e = (k*envfit.x2_ss*(s0-envfit.x2_ss)**2)
    f_e = 0.0 #relaxation
    f_ac = par[12]
  
  print('zeta ',zeta)
  print('del_zeta ',del_zeta)
  print('k ',k)
  print('delk ',delk)
  print('delm ',delm)
  print('f_e ',f_e)
  print('s0 ',s0)
  return (delm,1.5*k,delk,8*zeta,del_zeta,1.1*f_e,s0,f_ac)

class pol_envguessmodel(object):
  """ECO model parameters guessed from the envelop fit parameters of experimental image data"""
  def __init__(self, par,x0,envdata,envfit,numdrops=2):
    self.tol =[1.0e-8, 1.0e-6] # default values of [abserr, relerr]
    #self.envdata=envdata
    self.par = par
    self.numdrops=numdrops
    #self.envdata=envdata
    #self.envdata.x1min=eco.envelopfit_min(envdata.t1min,envdata.x1min)
    #self.envdata.x2min=eco.envelopfit_min(envdata.t2min,envdata.x2min)
    #self.envdata.x1max=eco.envelopfit_max(envdata.t1max,envdata.x1max)
    #self.envdata.x2max=eco.envelopfit_max(envdata.t2max,envdata.x2max)
    
    self.thcoef=envfit_to_guesscoeff(self.par,envfit,numdrops)
    (self.del_m,self.k,self.del_k,self.zeta,self.del_zeta,self.f_e,self.s0,self.f_ac) = self.thcoef

    self.t=np.arange(min(envdata.t1min[0],envdata.t1max[0]),envdata.t1max[len(envdata.t1max)-1],(envdata.t1max[len(envdata.t1max)-1]-envdata.t1max[0])/(20*len(envdata.t1max)))
    self.x0=x0 # can skip input and call initial guess estimation function
    self.x_envg= sol(self.t,self.thcoef,self.x0,self.tol) #guess solution
    self.x1_envg =self.x_envg[0]
    self.x2_envg =self.x_envg[2]

class pol_optfitmodel(object):
      """model fit coefficients and solution for a given exp data and theoretical/guess fit."""
      def __init__(self, guessmodel,imgdata):
        self.optiter= 3 # default number of iterations of global optimization=15, to be sure about the optimization
        #self.guessmodel = guessmodel
        self.t=guessmodel.t
        self.imgdata=imgdata
        self.thcoef = guessmodel.thcoef
        self.x0 = guessmodel.x0
        self.tol = guessmodel.tol
        self.numdrops = guessmodel.numdrops
        (self.del_m,self.k,self.del_k,self.zeta,self.del_zeta,self.f_e,self.s0,self.f_ac) = (guessmodel.del_m,guessmodel.k,guessmodel.del_k,guessmodel.zeta,guessmodel.del_zeta,guessmodel.f_e,guessmodel.s0,guessmodel.f_ac)
        self.optfitcoef = fit(self.optiter,self.imgdata,self.thcoef,self.x0,self.tol,self.numdrops)
        (self.c_k,self.c_delk,self.c_zeta,self.c_del_zeta,self.c_fe) = self.optfitcoef[0]
        self.resid=self.optfitcoef[1]
        self.resid0=self.optfitcoef[2]
        self.optcoef = (self.del_m,self.k*self.c_k,self.del_k*self.c_delk,self.zeta*self.c_zeta,self.del_zeta*self.c_del_zeta,self.f_e*self.c_fe,self.s0,self.f_ac)
        self.x_optfit= sol(self.t,self.optcoef,self.x0,self.tol) #Numerically fit solution
        self.x1_fit =self.x_optfit[0]
        self.x2_fit =self.x_optfit[2]
        for i in np.arange(1,len(self.t)-1): #BDF for velocity
          self.v1_fit = self.x1_fit[i]-self.x1_fit[i-1]
        #for i in np.arange(1,len(tdata)-1):
          self.v2_fit = self.x2_fit[i]-self.x2_fit[i-1]
        #for i in np.arange(2,len(self.t)-2): #BDF for acceleration
        #  self.a1_fit = self.v1_fit[i]-self.v1_fit[i-1]
        #for i in np.arange(1,len(self.t)-1):
        #  self.a2_fit = self.v2_fit[i]-self.v2_fit[i-1]

################### IV. Dynamic phase: Damping & Beats #################
#def finddynphase(imagedata,envdata_t1max,envdata_t1min,envdata_t2max,envdata_t2min,envdata_x1max,envdata_x1min,envdata_x2max,envdata_x2min,coef): # file loop above this call and plt.show() after the loop ends
def pol_finddynphase(imagedata,envdata,imgenvfit,coef): # file loop above this call and plt.show() after the loop ends
    print('\n Finding dynamic phase category...\n')
    c = coef[5]
    k = coef[1]
    s0 = coef[6]
    
    #T1=eco.envelopdata(imagedata).Tosc1
    #f_osc1= [1/i for i in T1]
    #omg_osc1=(2*np.pi())*f_osc1
    omg_osc1_avg= 2*np.pi/envdata.Tosc1_avg #Harmonic mean : np.average(np.delete(omg_osc1,0))
    #T2=eco.envelopdata(imagedata).Tosc2
    #f_osc2= [1/i for i in T2]
    #omg_osc1=(2*np.pi())*f_osc1
    omg_osc2_avg= 2*np.pi/(envdata.Tosc2_avg) # Harmonic mean = np.average(np.delete(omg_osc2,0))
    omg_b1=(imgenvfit.envfit1_max.omg_b+imgenvfit.envfit1_min.omg_b)/2
    omg_b2=(imgenvfit.envfit2_max.omg_b+imgenvfit.envfit2_min.omg_b)/2 #omg_b1=(envelopfit_min(envdata.t1max,envdata.x1max).omg_b+envelopfit_min(envdata.t1min,envdata.x1min).omg_b)/2
    tau_d1=(imgenvfit.envfit1_max.tau_d+imgenvfit.envfit1_min.tau_d)/2
    tau_d2=(imgenvfit.envfit2_max.tau_d+imgenvfit.envfit2_min.tau_d)/2
    
    ratio1 = omg_osc1_avg/omg_b1
    ratio2 = omg_osc2_avg/omg_b2

    xaxis=abs(2.0*(omg_osc1_avg-omg_osc2_avg)/(omg_osc1_avg+omg_osc2_avg))
    #xaxis = delm
    yaxis = c/(s0**3.0*k)#sqrt(c/(k*s0))/s0
    #yaxis = Ca_e

    if ratio1>=10. and ratio2>=10. :
        #print('damping with ratios osc/beats: ',ratio1, ratio2)
        dynphase  = 'damping'

    elif (ratio1<10. and ratio1>1.) or (ratio2<10. and ratio2 >1.):
        #print('Beats with ratios osc/beats: ',ratio1, ratio2)
        dynphase  = 'beats'
        
    elif ratio1<=1. and ratio2<=1. :
        #print('Chaotic oscillations with ratios osc/beats: ',ratio1, ratio2)
        dynphase  = 'chaotic_oscillations'

    else:
      dynphase  = 'chaotic & simply damping'
    # add condition for super critically damped
    print('category =',dynphase)
    return(dynphase, ratio1,ratio2,xaxis,yaxis,omg_osc1_avg,omg_osc2_avg,omg_b1,omg_b2)

class pol_dynphase(object):
  """Identify the dynphase category"""
  def __init__(self, imagedata,envdata,imgenvfit,coef):
    self.dynamicphase=finddynphase(imagedata,envdata,imgenvfit,coef)
    (self.dynphase, self.ratio1,self.ratio2,self.xaxis,self.yaxis,self.omg_osc1_avg,self.omg_osc2_avg,self.omg_b1,self.omg_b2)= self.dynamicphase