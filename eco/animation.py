"""
rendered deformation
evolving time series of x1,x2 vs t
evolving x1 vs x2 phase plot
moving point on displ vs velocity landscape - from model
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from matplotlib.gridspec import GridSpec
import pylab as pyl
#import matplotlib as mpl mpl.rcParams['animation.ffmpeg_path'] = r'C:\\Users\\ffmpeg\\bin\\ffmpeg.exe'
#from matplotlib.gridspec import GridSpec
#from matplotlib.animation import FuncAnimation 
#from matplotlib.animation import Animation 
#from matplotlib.animation import MovieWriter 
#from matplotlib.animation import FFMpegBase 
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
currentfolder = "E:/pramod_anchroedDrop/1_DC_damping_oscillations_analysis/output_single_drop_relax_compare_theory"
print('\n folder of animations output is at : ',currentfolder)

def render_drops_anim(t,x1,x2,theta_anch,radius1,radius2,separation,framerate,filename):
	t_range = len(t)
	#t_range = np.linspace(0.0 ,0.5,25)
	#theta_anch = 3*np.pi/4
	theta1 = np.linspace(theta_anch, -theta_anch, 200)
	theta2 = np.linspace(np.pi-theta_anch, 2*np.pi-(np.pi-theta_anch), 200)
	#theta = np.linspace(0, 2 * np.pi, 200)
	l2_1 =(3.0*(np.cos(theta1[:]))**2-1.0)/2.0
	l2_2 =(3.0*(np.cos(theta2[:]))**2-1.0)/2.0

	#fig = plt.figure(figsize=(12,6))
	fig, (ax0, ax1) = plt.subplots(1,2, subplot_kw=dict(projection='polar'), frameon=True)
	#fig.add_axes(frameon=True,linewidth=10, edgecolor="#04253a")
	#plt.figure(linewidth=10, edgecolor="#04253a")
	plt.subplots_adjust(wspace=0.0000)
	#gs = GridSpec(nrows=2, ncols=1)
	#axL = fig.add_subplot(gs[0, 0])
	#ax0 = plt.subplot(111, polar=True)
	ax0.set_rmax(radius1+separation/2)
	ax0.axis(False) 
	ax0.set_rticks([0.5, 1, 1.5, 2])  # Less radial ticks
	ax0.grid(False)
	#axR = fig.add_subplot(gs[1, 0])
	#ax1 = plt.subplot(111, polar=True)
	ax1.set_rmax(radius2+separation/2)
	ax1.axis(False) 
	ax1.set_rticks([0.5, 1, 1.5, 2])  # Less radial ticks
	ax1.grid(False)
	#ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
	

	drop1, = ax0.plot([],[])
	drop2, = ax1.plot([],[])
	time_template = 'time = %.1fs'
	time_lable1 = ax0.text(0.05, 0.9, '', transform=ax0.transAxes)
	#time_lable2 = ax1.text(0.05, 0.9, '', transform=ax1.transAxes)

	def update(b):
	    drop1.set_xdata(theta1)
	    drop2.set_xdata(theta2)
	    drop1.set_ydata(radius1+x1[b-1]*l2_1*100)
	    drop2.set_ydata(radius2+x2[b-1]*l2_2*100)

	    time_lable1.set_text(time_template % (b*1))
	    #time_lable2.set_text(time_template % (b*1))
	    return drop1,drop2, time_lable1

	#print('\n\n radius1 is : ', radius1)
	#print('\n\n radius2 is : ', radius2)
	ani = anim.FuncAnimation(fig, update, frames=t_range,interval=2e-8, blit=True, repeat=False)
	plt.draw()
	plt.show()
	print('saving rendered drops oscillations animation...')
	filespath = os.path.join(currentfolder, filename)
	if not os.path.exists(filespath):
		os.makedirs(filespath)
	ani.save(currentfolder+'/'+filename+'/'+'rendered_anim_'+filename+'.avi', fps=framerate/1000)#writer='imagemagick', writer = 'PillowWriter'

def anim_kineticsplot1(t1_adata,t2_adata,x1data,x2data,a1data,a2data,m,delm,framerate,filename): #render_anim(t,x1,x2,theta_anch,radius1,radius2,separation,framerate,filename)
	x1_adata = np.delete(x1data,0)
	t1_adata = np.delete(x1data,0)
	x2_adata = np.delete(x2data,0)
	t2_adata = np.delete(x2data,0)
	x1_adata = np.delete(x1_adata,0)
	t1_adata = np.delete(x1_adata,0)
	x2_adata = np.delete(x2_adata,0)
	t2_adata = np.delete(x2_adata,0)
	x1_adata = np.delete(x1_adata,-1)
	t1_adata = np.delete(x1_adata,-1)
	x1_adata = np.delete(x1_adata,-1)
	t1_adata = np.delete(x1_adata,-1)
	x1_adata = np.delete(x1_adata,-1)
	t1_adata = np.delete(x1_adata,-1)
	x2_adata = np.delete(x2_adata,-1)
	t2_adata = np.delete(x2_adata,-1)
	x2_adata = np.delete(x2_adata,-1)
	t2_adata = np.delete(x2_adata,-1)
	x2_adata = np.delete(x2_adata,-1)
	t2_adata = np.delete(x2_adata,-1)
	t_range= min(len(x1_adata),len(x2_adata)) #int(0.3*(min(len(x1_adata),len(x2_adata))))
	#(time1,time2)=([],[])

	fig = plt.figure(figsize=(10, 5)) # vertical stack
	gs = GridSpec(nrows=2, ncols=1)
	ax0 = fig.add_subplot(gs[0, 0])
	ax1 = fig.add_subplot(gs[1, 0])
	print('starting kin plot1_force..')
	#fig, (ax0, ax1) = plt.subplots(1,2) # horizontal stack

	
	ax0.set_title('net force vs x1') 
	ax0.legend('exp') #([time1])
	ax0.set_xlim([np.min(x1_adata[0:t_range]),np.max(x1_adata[0:t_range])])
	ax0.set_ylim([m*(1+delm)*np.min(np.asarray(a1data[0:t_range])),m*(1+delm)*np.max(np.asarray(a1data[0:t_range]))])

	ax1.set_title('net force vs x2')
	ax1.legend('exp') #([time2])
	ax1.set_xlim([np.min(x2_adata[0:t_range]),np.max(x2_adata[0:t_range])])
	ax1.set_ylim([m*(1+delm)*np.min(np.asarray(a2data[0:t_range])),m*(1+delm)*np.max(np.asarray(a2data[0:t_range]))])

	plt.suptitle ('net force vs displacement')

	drop1, = ax0.plot([],[])
	#ax0.plot([],[])
	drop2, = ax1.plot([],[])
	#ax1.plot([],[])
	#time_template = 'time = %.4fs'
	#time_lable1 = ax0.text(0.05, 0.9, '', transform=ax0.transAxes)
	#time_lable2 = ax1.text(0.05, 0.9, '', transform=ax1.transAxes)
	def update(b):
		drop1.set_xdata(x1_adata[0:b-1])
		drop2.set_xdata(x2_adata[0:b-1])
		drop1.set_ydata(m*(1+delm)*np.asarray(a1data[0:b-1]))
		drop2.set_ydata(m*(1-delm)*np.asarray(a2data[0:b-1]))
		#time1 = t1_adata[b-1]
		#time2 = t2_adata[b-1]
		return drop1,drop2#,time1,time2#ax0,ax1

	plt.draw()
	ani = anim.FuncAnimation(fig, update, frames=t_range,interval=5/t_range, blit=True, repeat=False)
	
	plt.show()
	print('saving force plot..')
	filespath = os.path.join(currentfolder, filename)
	if not os.path.exists(filespath):
		os.makedirs(filespath)
	ani.save(currentfolder+'/'+filename+'/'+'anim_kin1_'+filename+'.avi',fps=framerate/1000)#writer='imagemagick', writer = 'PillowWriter'

def anim_kineticsplot2(x1data,x2data,a1data,a2data,m,delm,framerate,filename): #energy vs displacement
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
	t_range= min(len(x1_adata),len(x2_adata))

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

	fig = plt.figure(figsize=(10, 5)) # vertical stack
	gs = GridSpec(nrows=2, ncols=1)
	ax0 = fig.add_subplot(gs[0, 0])
	ax1 = fig.add_subplot(gs[1, 0])
	print('starting kin plot2_energy ..')

	ax0.set_title('energy vs x1') 
	ax0.legend('exp')
	ax0.set_xlim([np.min(x1_adata[0:t_range]),np.max(x1_adata[0:t_range])])
	ax0.set_ylim([np.min(F1[0:t_range]),np.max(F1[0:t_range])])

	ax1.set_title('energy vs x2')
	ax1.legend('exp') 
	ax1.set_xlim([np.min(x2_adata[0:t_range]),np.max(x2_adata[0:t_range])])
	ax1.set_ylim([np.min(F2[0:t_range]),np.max(F2[0:t_range])])

	plt.suptitle ('Energy landscape')

	drop1, = ax0.plot([],[])
	drop2, = ax1.plot([],[])

	def update(b):
		drop1.set_xdata(x1_adata[0:b-1])
		drop2.set_xdata(x2_adata[0:b-1])
		drop1.set_ydata(np.asarray(F1[0:b-1]))
		drop2.set_ydata(np.asarray(F2[0:b-1]))
		return drop1,drop2

	plt.draw()
	ani = anim.FuncAnimation(fig, update, frames=t_range,interval=5/t_range, blit=True, repeat=False)
	
	plt.show()
	print('saving energy plot animation..')
	filespath = os.path.join(currentfolder, filename)
	if not os.path.exists(filespath):
		os.makedirs(filespath)
	ani.save(currentfolder+'/'+filename+'/'+'anim_kin2_'+filename+'.avi',fps=framerate/1000)

"""
def anim_tseries_fit (imgdata,envdata,envfit,modelfit,framerate,filename):
	t_range= min(len(x1_adata),len(x2_adata))

    ax1.scatter(imgdata.t2resol,imgdata.x2resol)
    ax1.scatter(envdata.t2max,envdata.x2max,label='_nolegend_')
    ax1.scatter(envdata.t2min,envdata.x2min,label='_nolegend_')
    ax1.plot(envfit.envfit2_max.t_env_fit,envfit.envfit2_max.x_env_fit)
    ax1.plot(envfit.envfit2_min.t_env_fit,envfit.envfit2_min.x_env_fit)
    ax1.plot(modelfit.t,modelfit.x2_manual,'-b')
    ax1.xaxis.set_major_locator(MultipleLocator(0.005))
    ax1.xaxis.set_minor_locator(MultipleLocator(0.001))
    ax1.set_title('x2 data')
    ax1.legend(['expt','max envelop fit','min envelop fit','ECO fit'])

    plt.suptitle ('x1 and x2 exp & ECO model of : \n'+filename)
    plt.draw()

	#(time1,time2)=([],[])

	fig = plt.figure(figsize=(10, 5)) # vertical stack
	gs = GridSpec(nrows=2, ncols=1)
	ax0 = fig.add_subplot(gs[0, 0])
	ax1 = fig.add_subplot(gs[1, 0])
	print('starting kin plot')
	#fig, (ax0, ax1) = plt.subplots(1,2) # horizontal stack

	
	ax0.scatter(imgdata.t1resol,imgdata.x1resol,label='exp')
    ax0.scatter(envdata.t1max,envdata.x1max,label='_nolegend_')
    ax0.scatter(envdata.t1min,envdata.x1min,label='_nolegend_')
    ax0.plot(envfit.envfit1_max.t_env_fit,envfit.envfit1_max.x_env_fit)
    ax0.plot(envfit.envfit1_min.t_env_fit,envfit.envfit1_min.x_env_fit)
    ax0.plot(modelfit.t,modelfit.x1_manual,'-b')
    ax0.xaxis.set_major_locator(MultipleLocator(0.005))
    ax0.xaxis.set_minor_locator(MultipleLocator(0.001))
    ax0.set_title('x1 data') 
    ax0.legend(['expt','max envelop fit','min envelop fit','ECO fit'])
	ax0.set_xlim([np.min(imgdata.t1resol[0:t_range]),np.max(imgdata.t1resol[0:t_range])])
	ax0.set_ylim([np.min(imgdata.x1resol[0:t_range]),np.max(imgdata.x1resol[0:t_range])])

	ax1.set_title('net force vs x2')
	ax1.legend('exp') #([time2])
	ax1.set_xlim([np.min(x2_adata[0:t_range]),np.max(x2_adata[0:t_range])])
	ax1.set_ylim([m*(1+delm)*np.min(np.asarray(a2data[0:t_range])),m*(1+delm)*np.max(np.asarray(a2data[0:t_range]))])

	plt.suptitle ('net force vs displacement')

	drop1, = ax0.plot([],[])
	#ax0.plot([],[])
	drop2, = ax1.plot([],[])
	#ax1.plot([],[])
	#time_template = 'time = %.4fs'
	#time_lable1 = ax0.text(0.05, 0.9, '', transform=ax0.transAxes)
	#time_lable2 = ax1.text(0.05, 0.9, '', transform=ax1.transAxes)
	def update(b):
		drop1.set_xdata(x1_adata[0:b-1])
		drop2.set_xdata(x2_adata[0:b-1])
		drop1.set_ydata(m*(1+delm)*np.asarray(a1data[0:b-1]))
		drop2.set_ydata(m*(1-delm)*np.asarray(a2data[0:b-1]))
		#time1 = t1_adata[b-1]
		#time2 = t2_adata[b-1]
		return drop1,drop2#,time1,time2#ax0,ax1

	plt.draw()
	ani = anim.FuncAnimation(fig, update, frames=t_range,interval=5/t_range, blit=True, repeat=False)
	
	plt.show()
	print('saving..')
	filespath = os.path.join(currentfolder, filename)
	if not os.path.exists(filespath):
		os.makedirs(filespath)
	ani.save(currentfolder+'/'+filename+'/'+'anim_kin1'+filename+'.avi',fps=framerate/1000)#writer='imagemagick', writer = 'PillowWriter'
""" 

def anim_phaseplot1(x1data,x2data,framerate,filename):
	t_range= len(x1data) #int(0.3*len(x1_adata))
	#(time1,time2)=([],[])
	x1data = np.asarray(x1data)
	x2data = np.asarray(x2data)

	fig = plt.figure(figsize=(10, 5)) # vertical stack
	gs = GridSpec(nrows=2, ncols=1)
	ax0 = fig.add_subplot(gs[0, 0])
	ax1 = fig.add_subplot(gs[1, 0])
	print('\nstarted phaseplot1 animation \n') #, len(x1data), len(x2data))
	#fig, (ax0, ax1) = plt.subplots(1,2) # horizontal stack
	
	ax0.set_title('x2 vs x1') 
	ax1.set_title('x2 vs x1') 
	ax0.legend('exp') #([time1])
	ax1.legend('exp') #([time1])
	#ax0.xlim([np.min(x1data[0:t_range]),np.max(x1data[0:t_range])])
	ax0.set_xlim([np.min(x1data[0:t_range]),np.max(x1data[0:t_range])])
	#ax0.ylim([np.min(np.asarray(x2data[0:t_range])),np.max(np.asarray(x2data[0:t_range]))])
	ax0.set_ylim([np.min(np.asarray(x2data[0:t_range])),np.max(np.asarray(x2data[0:t_range]))])

	drop1, = ax0.plot([],[])
	drop2, = ax1.plot([],[])
	#time_template = 'time = %.4fs'
	#time_lable1 = ax0.text(0.05, 0.9, '', transform=ax0.transAxes)
	#time_lable2 = ax1.text(0.05, 0.9, '', transform=ax1.transAxes)
	def update(b):
		drop1.set_xdata(x1data[0:b+2])
		drop2.set_xdata(x1data[0:b+2])
		drop1.set_ydata(x2data[0:b+2])
		drop2.set_ydata(x2data[0:b+2])
		#time1 = t1_adata[b-1]
		return drop1,drop2

	plt.draw()
	ani = anim.FuncAnimation(fig, update, frames=t_range,interval=1/t_range, blit=True, repeat=False)
	
	#plt.show()
	print('saving...\n')
	filespath = os.path.join(currentfolder, filename)
	if not os.path.exists(filespath):
		os.makedirs(filespath)
	ani.save(currentfolder+'/'+filename+'/'+'anim_phplot1_'+filename+'.avi',fps=framerate/1000)#writer='imagemagick', writer = 'PillowWriter'

def anim_phaseplot2(x1data,x2data,v1data,v2data,framerate,filename):
	x1_vdata = np.delete(x1data,0)
	x1_vdata = np.delete(x1_vdata,-1)
	x2_vdata = np.delete(x2data,0)
	x2_vdata = np.delete(x2_vdata,-1)
	t_range= min(len(x1_vdata),len(x2_vdata))

	fig = plt.figure(figsize=(10, 5)) # vertical stack
	gs = GridSpec(nrows=2, ncols=1)
	ax0 = fig.add_subplot(gs[0, 0])
	ax1 = fig.add_subplot(gs[1, 0])
	print('\nstarted phplot2 animation\n')

	ax0.set_title('v1 vs x1') 
	ax0.legend('exp') #([time1])
	ax0.set_xlim([np.min(x1_vdata[0:t_range]),np.max(x1_vdata[0:t_range])])
	ax0.set_ylim([np.min(v1data[0:t_range]),np.max(v1data[0:t_range])])

	ax1.set_title('v2 vs x2')
	ax1.legend('exp') #([time2])
	ax1.set_xlim([np.min(x2_vdata[0:t_range]),np.max(x2_vdata[0:t_range])])
	ax1.set_ylim([np.min(v2data[0:t_range]),np.max(v2data[0:t_range])])

	plt.suptitle ('velocity vs displacement')

	drop1, = ax0.plot([],[])
	drop2, = ax1.plot([],[])

	def update(b):
		drop1.set_xdata(x1_vdata[0:b-1])
		drop2.set_xdata(x2_vdata[0:b-1])
		drop1.set_ydata(v1data[0:b-1])
		drop2.set_ydata(v2data[0:b-1])
		#time1 = t1_adata[b-1]
		#time2 = t2_adata[b-1]
		return drop1,drop2#,time1,time2#ax0,ax1

	plt.draw()
	ani = anim.FuncAnimation(fig, update, frames=t_range,interval=1/t_range, blit=True, repeat=False)
	
	#plt.show()
	print('saving...\n')
	filespath = os.path.join(currentfolder, filename)
	if not os.path.exists(filespath):
		os.makedirs(filespath)
	ani.save(currentfolder+'/'+filename+'/'+'anim_phplot2_'+filename+'.avi',fps=framerate/1000)

def anim_polarplot(theta1_data,theta2_data,x1_theta,x2_theta,framerate,filename): #not working well
	frame_range= min(len(x1_theta),len(x2_theta))
	print('started polar plot animation')

	#print('\n\n\n lengths: ',len(theta1_data),len(theta2_data),'\n\n\n')
	#fig = plt.figure(figsize=(10,5))
	fig, (ax0, ax1) = plt.subplots(1,2, subplot_kw=dict(projection='polar'))
	#gs = GridSpec(nrows=2, ncols=1)
	#ax0 = fig.add_subplot(gs[0, 0])
	#ax0 = fig.add_subplot(111, projection='polar')
	#ax1 = fig.add_subplot(gs[1, 0])
	#ax1 = fig.add_subplot(111, projection='polar')
	#ax0.plot(theta1_data, x1_theta)

	ax0.set_ylim(0,np.max(x1_theta[0:frame_range]))
	#ax0.set_rmax(np.max(x1_theta[0:frame_range]))
	#ax0.set_rmin(0)
	#ax0.plot(theta1_data, x1_theta)
	#ax0.set_rticks([0.1*max(x1_theta[0:frame_range]), 0.25*max(x1_theta[0:frame_range]), 0.5*max(x1_theta[0:frame_range]), 1.0*max(x1_theta[0:frame_range])])  # Less radial ticks
	ax0.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
	ax0.grid(True)
	ax0.legend(['drop1'])

	ax1.set_ylim(0,np.max(x2_theta[0:frame_range]))
	#ax1.set_rmax(np.max(x2_theta[0:frame_range]))
	#ax1.set_rmin(0)
	#ax1.plot(theta2_data, x2_theta)
	#ax1.set_rticks([0.1*max(x2_theta[0:frame_range]), 0.25*max(x2_theta[0:frame_range]), 0.5*max(x2_theta[0:frame_range]), 1.0*max(x2_theta[0:frame_range])])  # Less radial ticks
	ax1.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
	ax1.grid(True)
	ax1.legend(['drop2'])

	plt.suptitle("Radial plot of displacement vs phase angle of individual oscillations", va='bottom')

	drop1, = ax0.plot([],[])
	drop2, = ax1.plot([],[])

	def update(b):
		#print(b)
		drop1.set_xdata(theta1_data[0:b])
		drop2.set_xdata(theta2_data[0:b])
		drop1.set_ydata(x1_theta[0:b])
		drop2.set_ydata(x2_theta[0:b])
		#print(theta1_data[b],'\n\n\n\n',x1_theta[b],'\n\n\n\n')
		#time_lable1.set_text(time_template % (b*1))
		return drop1,drop2
	plt.draw()
	ani = anim.FuncAnimation(fig, update, frames=frame_range,interval=1/frame_range, blit=True, repeat=False)
	
	#plt.show()
	print('saving...')
	filespath = os.path.join(currentfolder, filename)
	if not os.path.exists(filespath):
		os.makedirs(filespath)
	ani.save(currentfolder+'/'+filename+'/'+'anim_polarplot_'+filename+'.avi',fps=framerate/1000)