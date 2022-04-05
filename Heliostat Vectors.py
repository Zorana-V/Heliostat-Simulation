# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 14:33:25 2020

@author: brigh
"""


#from datetime import datetime,date,time,timedelta
import datetime
import time
import math
#from geopy.geocoders import Nominatim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy.optimize as sci
from PIL import Image,ImageFilter
import pandas as pd
import openpyxl as ex2m


#elapsed time
start = time.time()

#To get day in a year
today = datetime.datetime.now()
month = 6 #Month of Interest
day = 20 #Day of Interest
n = (datetime.datetime(today.year, month, day) - datetime.datetime(today.year, 1, 1)).days + 1
months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
print ('\nDay = ')
print(n)

#Input information about location and Heliostat
#Gainesville Florida
phi = math.radians(29.6515)
Lloc = math.radians(82.3248)
timezone = 5
Lst = math.radians(15*timezone)
height = 10 #height of tower(ft)
distance = 20 #distance between tower and surface (ft)
az_t = np.linspace(-math.pi/2, math.pi/2, 31) #azumith angle of tower (raidans)
hel = [15] #Select Heliostat(s) of interest for angle graphs
c = hel[0] #Select one Heliostat for animation

#Calculating Time vector
delinc = math.radians(23.45)*math.sin(math.radians(360*((284+n)/365)))
B = math.radians((n-1)*(360/365))
E = 229.2*(0.000075 + 0.001868*math.cos(B) - 0.032077*math.sin(B) - 0.014615*math.cos(2*B) - 0.04089*math.sin(2*B))
w_set = math.acos(-math.tan(phi)*math.tan(delinc))
w_rise = -w_set
sunset = 12+(w_set/(math.radians(15)))
sunrise = 12+(w_rise/(math.radians(15)))
Time = np.linspace(sunrise, sunset, 144)

#initializing vectors
theta = np.zeros((len(Time),len(az_t)))
Beta = np.zeros((len(Time),len(az_t)))
azumith = np.zeros((len(Time),len(az_t)))
alt_n = np.zeros((len(Time),len(az_t)))
az_n = np.zeros((len(Time),len(az_t)))
H_x = np.zeros((len(Time),len(az_t)))
H_y = np.zeros((len(Time),len(az_t)))
H_z = np.zeros((len(Time),len(az_t)))
S_x = np.zeros((len(Time),len(az_t)))
S_y = np.zeros((len(Time),len(az_t)))
S_z = np.zeros((len(Time),len(az_t)))
n_x = np.zeros((len(Time),len(az_t)))
n_y = np.zeros((len(Time),len(az_t)))
n_z = np.zeros((len(Time),len(az_t)))

###### HELIOSTAT ORIENTATION #######
for x in range(len(Time)):
    for y in range(len(az_t)):
        w = math.radians((Time[x]-12)*15)
        thetaz = math.acos((math.sin(phi)*math.sin(delinc)) + (math.cos(phi)*math.cos(delinc)*math.cos(w)))
    
        #finding H
        alt_t = math.atan(height/distance)
        H_x[x][y] = math.cos(alt_t)*math.sin(az_t[y])
        H_y[x][y] = -math.cos(alt_t)*math.cos(az_t[y])
        H_z[x][y] = math.sin(alt_t)
        H = np.array([[H_x[x][y]], [H_y[x][y]], [H_z[x][y]]])
    
        #finding S
        alt_s = (math.pi/2) - thetaz
        az_s = (np.sign(w)*(-1))*(math.acos(((math.sin(delinc)*math.cos(phi))-(math.cos(w)*math.cos(delinc)*math.sin(phi)))/math.sin(thetaz)))
        S_x[x][y] = math.cos(alt_s)*math.sin(az_s)
        S_y[x][y] = -math.cos(alt_s)*math.cos(az_s)
        S_z[x][y] = math.sin(alt_s)
        S = np.array([S_x[x][y], S_y[x][y], S_z[x][y]])

        #calculating angle of incidence and slope
        the = math.acos((np.dot(S,H)))/2
        theta[x][y] = math.degrees(the)
        
        
        def nSolver(n):
            n_1 = n[0]
            n_2 = n[1]
            n_3 = n[2]
            
            F = np.zeros(3)
            F[0] = (H_y[x][y]*S_z[x][y] - H_z[x][y]*S_y[x][y])*n_1 + (H_z[x][y]*S_x[x][y] - H_x[x][y]*S_z[x][y])*n_2 + (H_x[x][y]*S_y[x][y] - H_y[x][y]*S_x[x][y])*n_3
            F[1] = H_x[x][y]*n_1 + H_y[x][y]*n_2 + H_z[x][y]*n_3 - math.cos(the)
            F[2] = np.sqrt(pow(n_1,2) + pow(n_2,2) + pow(n_3,2)) - 1
            return F
        if x == 0:
            nguess = np.array([0,0,0])
        else:
            nguess = np.array([n_x[x-1][y],n_y[x-1][y],n_z[x-1][y]])
        n = sci.fsolve(nSolver,nguess)
        n_x[x][y] = n[0]
        n_y[x][y] = n[1]
        n_z[x][y] = n[2]
        
        alt = math.asin(n_z[x][y])
        az = math.asin(n_x[x][y]/math.cos(alt))
        alt_n[x][y] = math.degrees(alt)
        az_n[x][y] = math.degrees(az)
        
        Beta[x][y] = 90 - math.degrees(alt)
        azumith[x][y] = math.degrees(az)

###### SOLAR RADIATION #######
Gsc = 1367 #W m-2

#Aluminum Data
A = pd.read_excel('Alum.xlsx') 
A = pd.DataFrame.to_numpy(A)
r = len(A)
lamA = A[range(1,r-1),2] #wavelength
PerA = A[range(1,r-1),5] #Percent Reflectivity

#Silver Economy Data
SiE = pd.read_excel('Silver_Economy.xlsx') 
SiE = pd.DataFrame.to_numpy(SiE)
lamSiE = SiE[range(1,r-1),2] #wavelength
PerSiE = SiE[range(1,r-1),3] #Percent Reflectivity

#Silver Data
r = range(2,1979)
Si = pd.read_excel('Silver.xlsx') 
Si = pd.DataFrame.to_numpy(Si)
lamSi = Si[r,4] #wavelength
PerSi = Si[r,5] #Percent Reflectivity


#Import Extraterrestrial Solar Irradiance Data
ESI = pd.read_excel('ESI.xlsx')
ESI = pd.DataFrame.to_numpy(ESI)
wl = ESI[:,0] #wavelength
f = ESI[:,1] #fraction


ReIrrA = np.zeros(len(wl))
ReIrrSi = np.zeros(len(wl))
ReIrrSiE = np.zeros(len(wl))
Rad = np.zeros(len(wl))
#CALCULATE TOTAL IRRADIANCE
for x in range(1,len(wl)):
    Rad[x] = (Gsc*f[x-1]*(wl[x]-wl[x-1]))
    for y in range(len(lamA)):
        if (wl[x] - lamA[y] <= 0.005):
            ReIrrA[x] = (Gsc*f[x-1]*(wl[x]-wl[x-1])*(PerA[y]/100))
            
        if (wl[x] - lamSi[y] <= 0.005):
            ReIrrSi[x] = (Gsc*f[x-1]*(wl[x]-wl[x-1]))*(PerSi[y]/100)
        
        if (wl[x] - lamSiE[y] <= 0.005):
            ReIrrSiE[x] = (Gsc*f[x-1]*(wl[x]-wl[x-1]))*(PerSiE[y]/100)

ReA = sum(ReIrrA)
ReSi = sum(ReIrrSi)
ReSiE = sum(ReIrrSiE)
ActRad = sum(Rad)

eA = ReA/ActRad
eSi = ReSi/ActRad
eSiE = ReSiE/ActRad

##PLOT THEM
#plt.rcParams.update({'font.size': 15})
plt.rcParams.update({'font.size': 15})
plt.figure("""Mirror Type % Reflectance""")
plt.title('Percentage of Reflectance of Each Mirror Type')
plt.xlabel('Wavelength (microm)')
plt.ylabel('Percentage Reflectance (%)')
plt.xlim([0.3,2.0])
plt.ylim([75,100])
plt.plot(lamA,PerA, label = 'Aluminum')
plt.plot(lamSi,PerSi,label = 'Silver')
plt.plot(lamSiE,PerSiE, label = 'Silver Economy')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


###### GRAPHING #######
plt.rcParams.update({'font.size': 15})
plt.figure("""Slope (Beta)""")
plt.title('Slope of %d Heliostat(s) in Gainesville, FL on %s %i' % (len(hel),months[month-1],day))
plt.xlabel('Time (Hours)')
plt.ylabel('Slope (Degrees)')
for i in hel:
    az = math.degrees(az_t[i])
    plt.plot(Time,Beta[:,i], label = 'Tower Azumith = %s' % az)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

plt.figure("""Azumith (Gamma)""")
plt.title('Azumith Angle of %d Heliostat(s) in Gainesville, FL on %s %i' % (len(hel),months[month-1],day))
plt.xlabel('Time (Hours)')
plt.ylabel('Slope (Degrees)')
for i in hel:
    az = math.degrees(az_t[i])
    plt.plot(Time,azumith[:,i], label = 'Tower Azumith = %s' % az)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show() 

plt.figure("""Angle of Incidence (Theta)""")
plt.title('Angle of Incidence on %d Heliostat(s) in Gainesville, FL on %s %i' % (len(hel),months[month-1],day))
plt.xlabel('Time (Hours)')
plt.ylabel('Angle (Degrees)')
for i in hel:
    az = math.degrees(az_t[i])
    plt.plot(Time,theta[:,i], label = 'Tower Azumith = %s' % az)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

line = np.linspace(0,5,16)


plt.rcParams.update({'font.size': 40})
for n in range(len(Time)):
    az = math.degrees(az_t[c])
    fig = plt.figure(figsize=(40,20))
    ax = plt.axes(projection='3d')
    ax.set_xlim3d(2,-2)
    ax.set_ylim3d(-2,2)
    ax.set_zlim3d(-2,2)
    plt.title('Heliostat Rotation for Tower Azumith = %d on %s %s' % (az, months[month],day))
    ax.set_xlabel('South')
    ax.set_ylabel('East')
    Hx = H_x[n][c]*line
    Hy = H_y[n][c]*line
    Hz = H_z[n][c]*line
    Sx = S_x[n][c]*line
    Sy = S_y[n][c]*line
    Sz = S_z[n][c]*line
    nx = n_x[n][c]*line
    ny = n_y[n][c]*line
    nz = n_z[n][c]*line
    point = np.array([0, 0, 0])
    normal = np.array([n_x[n][c], n_y[n][c], n_z[n][c]])
    d = -point.dot(normal)
    xx, yy = np.meshgrid([-0.5,0.5],[-0.5,0.5])
    z = (-normal[0]*xx - normal[1]*yy - d)*1./normal[2]
    ax.plot3D(Hx,Hy,Hz,color='gray',label='Heliostat')
    ax.plot3D(Sx,Sy,Sz,color='orange',label='Sun')
    ax.plot3D(nx,ny,nz,color='blue',label='normal')
    ax.plot_surface(xx, yy, z,color='grey',alpha=0.2)
    plt.legend(loc=1)
    plt.savefig(str(n)+'.png')
    plt.close()
    
images = []
for n in range(len(Time)):
    exec('a'+str(n)+'=Image.open("'+str(n)+'.png")')
    images.append(eval('a'+str(n)))
images[0].save('Heliostat.gif', save_all=True, append_images=images[1:n], duration=5, loop=0)

e_time = time.time() - start
print('Elapsed Time: %d seconds' % e_time)

            