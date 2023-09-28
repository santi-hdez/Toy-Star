#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 16:15:10 2023

@author: santiago
"""

"""
                                                                              
 This procedure simulates a polytropic, 3D star model. We choose a sphere with radius R and mass content M. We place
 particles with mass M/N randomly within this sphere and integrate the hydro equations using the SPH algorithm under
 consideration of a damping force and a simplified gravitational term. 
 
 If run with the default values, the following will be observed during the simulation:
     
   1. Since the initial particle distribution is not in hydrostatic equilibrium, the whole particle distribution will
      expand.
   2. The particles will execute a damped oscillation around the final equilibrium state due to the attracting
      gravitational term lambd*x and the linear damping term damp*v.
 
 
 Execution: toy_star(N, R, M, n, k, damp, t0, tend, h, CFL, data_save, path_save)

 Input:

  N:           Number of simulated particles.
  R:           Sphere's radio.
  M:           Total mass. Particle's mass defined by mass=M/N.
  n:           Politropic index.
  k:           Politropic constant.
  damp:        Damping coefficient for the damping force.
  t0:          Initial simulation time.
  tend:        Final simulation time.
  h:           Smoothing length for the kernel function.
  CFL:         Courant number. Security factor <1 used for determining the time step by means of the Courant criterion.
  data_save:   Boolean parameter. If True, the execution of the code saves the coordinates and velocities of particles
               in a file.dat for each time step. If False, the execution does not save any data.
  path_save:   Path where the output of the simulation should be saved.
    

 Output:    
                                            
  It generates a new folder called Output in the indicated path by 'path_save'. If 'data_save' is True, the coordinates and
  velocities of particles are saved in a file.dat for each time step. Additionally, inside the Output folder, two folders
  called Movie1 and Movie2 are created where videos with the obtained simulation are stored.                                               

"""


def toy_star(N=1000, R=0.75, M=2.0, n=1.0, k=0.1, damp=0.1, t0=0.0, tend=40.0, h=0.2, CFL=0.5, data_save=True, path_save='/home/santiago/Documentos/Projects/Toy_Star'):
    
    import numpy as np
    from scipy.special import gamma
    import subprocess
    import os
    import shutil
    import matplotlib.pyplot as plt
    
    FileNames = []
    File_Names1 = []
    File_Names2 = []
    
    fpath1 = path_save+'/Output'
    fpath2 = fpath1+'/Movie1'
    fpath3 = fpath1+'/Movie2'

    os.mkdir(fpath1)
    os.mkdir(fpath2)
    os.mkdir(fpath3)
    
    gamma_cons = 1+1/n
        
    lambd = 2*k*(1+n)*np.pi**(-3/(2*n))*(M*gamma(2.5+n)/(R**3*gamma(1+n)))**(1./n)*R**(-2)
    
    mass = M/N
    
    #Initial SPH particle distribution:
    
    x = np.zeros(N)
    y = np.zeros(N)
    z = np.zeros(N)
    vx = np.zeros(N)
    vy = np.zeros(N)
    vz = np.zeros(N)
    m = np.zeros(N)
    
    i = 0
    
    while i < N:
        
        phi = np.random.uniform(high=2.*np.pi)
        costheta = np.random.uniform(low=-1.0, high=1.0)
        u = np.random.uniform()
        theta = np.arccos(costheta)
        r = R*np.cbrt(u)
    
        x[i] = r*np.sin(theta)*np.cos(phi)
        y[i] = r*np.sin(theta)*np.sin(phi)
        z[i] = r*np.cos(theta)
    
        m[i] = mass
        
        i += 1
    
    #Pressure computation:
    
    def pressure(k, gamma_cons, rho):
        
        p = k*rho**gamma_cons
        
        return p
    
    #Sound speed computation:
    
    def sound_speed(k, gamma_cons, rho):
        
        cs = np.sqrt(gamma_cons*k*rho**(gamma_cons-1))
        
        return cs
        
    #Finding interactive particles for each particle:
    
    def get_interactions(x, y, z, h):
        
        interactions = []
        
        for i in range(0, len(x)):
            
            indexlist = []
            
            for j in range(0, len(x)):
                
                if i == j:
                    
                    continue
                    
                rr = (x[i]-x[j])**2+(y[i]-y[j])**2+(z[i]-z[j])**2
                
                if np.sqrt(rr) < h:
                    
                    indexlist.append(j)
                    
            interactions.append(indexlist)
                    
        return interactions
    
    #Kernel function (cubic B-spline):
    
    def kernel(i, j, x, y, z, h, flag='3D'):
        
        if flag == '1D':
            
            sigma = 4/3  
            h_power = h
            h_power_2 = h**2
            
        if flag == '2D':
            
            sigma = 40/(7*np.pi)
            h_power = h**2
            h_power_2 = h**3
            
        if flag == '3D':
            
            sigma = 8/np.pi 
            h_power = h**3
            h_power_2 = h**4
        
        dx = x[i]-x[j]
        dy = y[i]-y[j]
        dz = z[i]-z[j]
        
        rr = dx**2+dy**2+dz**2
        
        if np.sqrt(rr)/h >= 0 and np.sqrt(rr)/h < 1/2:
    
            w_temp = 6*((np.sqrt(rr)/h)**3)-6*((np.sqrt(rr)/h)**2)+1
            dw = 3*((np.sqrt(rr)/h)**2)-2*(np.sqrt(rr)/h)
    
        elif np.sqrt(rr)/h >= 1/2 and np.sqrt(rr)/h < 1:
    
            w_temp = 2*(1-(np.sqrt(rr)/h))**3
            dw = -(1-(np.sqrt(rr)/h))**2
    
        elif np.sqrt(rr)/h > 1:
    
            w_temp = 0.0
            dw = 0.0
            
        w = (sigma/h_power)*w_temp        
        dwdx = ((6*sigma)/h_power_2)*(dx/np.sqrt(rr))*dw
        dwdy = ((6*sigma)/h_power_2)*(dy/np.sqrt(rr))*dw
        dwdz = ((6*sigma)/h_power_2)*(dz/np.sqrt(rr))*dw
        
        return w, dwdx, dwdy, dwdz
           
    #Density computation by means of the kernel function:
    
    def density(interactions, x, y, z, m, h, flag='3D'):
        
        rho = np.zeros(N)
        
        for i in range(0, len(rho)):
            
            for j in interactions[i]:
                
                w, dwdx, dwdy, dwdz = kernel(i, j, x, y, z, h, flag='3D')
                
                rho[i] += m[j]*w
                
        return rho
        
    #Accelerations computation:
    
    def accelerations(interactions, x, y, z, vx, vy, vz, m, rho, p, h, lambd, damp, flag='3D'):
        
        ax = np.zeros(N)
        ay = np.zeros(N)
        az = np.zeros(N)
        
        for i in range(0, len(x)):
            
            for j in interactions[i]:
                
                w, dwdx, dwdy, dwdz = kernel(i, j, x, y, z, h, flag='3D')
                
                ax[i] -= m[j]*(p[i]/rho[i]**2+p[j]/rho[j]**2)*dwdx 
                ay[i] -= m[j]*(p[i]/rho[i]**2+p[j]/rho[j]**2)*dwdy 
                az[i] -= m[j]*(p[i]/rho[i]**2+p[j]/rho[j]**2)*dwdz 
                
            ax[i] -= lambd*x[i]
            ay[i] -= lambd*y[i]
            az[i] -= lambd*z[i]
            
            ax[i] -= damp*vx[i]
            ay[i] -= damp*vy[i]
            az[i] -= damp*vz[i]
            
        return ax, ay, az
            
    #Time step computation (Courant criterion):
    
    def time_step(cs, h, CFL):
        
        Dt = CFL*(h/np.max(cs))
        
        return Dt
    
    #2nd order integrator (Verlet method (Leapfrog variant)): 
    
    def integrator(x, y, z, vx, vy, vz, Dt):
            
        interactions = get_interactions(x, y, z, h)
        rho = density(interactions, x, y, z, m, h, flag='3D')
        cs = sound_speed(k, gamma_cons, rho)
        p = pressure(k, gamma_cons, rho)
        ax, ay, az = accelerations(interactions, x, y, z, vx, vy, vz, m, rho, p, h, lambd, damp, flag='3D')
        Dt = time_step(cs, h, CFL)
        
        print("-", flush=True)
        
        vxr = vx[:]+0.5*Dt*ax
        vyr = vy[:]+0.5*Dt*ay
        vzr = vz[:]+0.5*Dt*az
        
        x += vxr*Dt
        y += vyr*Dt
        z += vzr*Dt
        
        interactions = get_interactions(x, y, z, h)
        rho = density(interactions, x, y, z, m, h, flag='3D')
        cs = sound_speed(k, gamma_cons, rho)
        p = pressure(k, gamma_cons, rho)
        ax, ay, az = accelerations(interactions, x, y, z, vx, vy, vz, m, rho, p, h, lambd, damp, flag='3D')
        Dt = time_step(cs, h, CFL)
        
        vx = vxr[:]+0.5*Dt*ax
        vy = vyr[:]+0.5*Dt*ay
        vz = vzr[:]+0.5*Dt*az
        
        return x, y, z, vx, vy, vz
    
    #Main loop:     
           
    t = t0
    it = 0.0
    
    while t < tend:
        
        interactions = get_interactions(x, y, z, h)
        
        rho = density(interactions, x, y, z, m, h, flag='3D')
        
        cs = sound_speed(k, gamma_cons, rho)
        
        p = pressure(k, gamma_cons, rho)
        
        ax, ay, az = accelerations(interactions, x, y, z, vx, vy, vz, m, rho, p, h, lambd, damp, flag='3D')
        
        Dt = time_step(cs, h, CFL)
        
        x, y, z, vx, vy, vz = integrator(x, y, z, vx, vy, vz, Dt)
        
        if data_save == True:
            
            # Save data in each timestep to a .dat file:
            
            output = np.zeros(x.size, dtype=[('var1', float), ('var2', float), ('var3', float), ('var4', float), ('var5', float), ('var6', float)])
            
            output['var1'] = x
            output['var2'] = y
            output['var3'] = z
            output['var4'] = vx
            output['var5'] = vy
            output['var6'] = vz
            
            filename = 'Data_t='+str({round(t,4)})+'.dat'
            FileNames.append(filename)
            
            np.savetxt(filename, output, fmt="%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f", header='X\tY\tZ\tVX\tVY\tVZ', comments=' ')
            
            shutil.move(filename, fpath1+'/'+filename)
        
        #Images creation for each timestep:
        
        fig, ax = plt.subplots(1,3)
        fig.subplots_adjust(wspace = 0.7)
        
        for _ax in ax:
            
            _ax.set_aspect('equal')
            _ax.set_xlim(-1,1)
            _ax.set_ylim(-1,1)
    
        ax[0].scatter(x, y, s=1)
        ax[1].scatter(x, z, s=1)
        ax[2].scatter(y, z, s=1)
        
        ax[0].set_xlabel('x')
        ax[0].set_ylabel('y')
        ax[1].set_xlabel('x')
        ax[1].set_ylabel('z')
        ax[2].set_xlabel('y')
        ax[2].set_ylabel('z')
        
        file_name1 = "Image_toy_star_{:.7f}.png".format(it)
        File_Names1.append(fpath2+'/'+file_name1) 
        fig.savefig(file_name1)
        plt.close()
        
        fig=plt.figure(figsize=(10,10))
        ax=fig.add_subplot(111,projection='3d')
        ax.set_xlim(-1,1)
        ax.set_ylim(-1,1)
        ax.set_zlim(-1,1)
        ax.scatter(x,y,z, s=40, marker = '.')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        file_name2 = "Image_toy_star_3D_{:.7f}.png".format(it)
        File_Names2.append(file_name2) 
        fig.savefig(fpath3+'/'+file_name2)
        plt.close()
        
        
        t += Dt
        it += 0.0000001
    
    #Use ffmpeg to combine the images into a movie:
    
    subprocess.run(['ffmpeg','-framerate','20','-pattern_type','glob','-i',fpath2+"/*.png",'-c:v','libx264','-pix_fmt','yuv420p','movie_toy_star.mp4'])
    shutil.move('movie_toy_star.mp4', fpath2+'/movie_toy_star.mp4')
    
    subprocess.run(['ffmpeg','-framerate','20','-pattern_type','glob','-i',fpath3+"/*.png",'-c:v','libx264','-pix_fmt','yuv420p','movie_toy_star_3D.mp4'])
    shutil.move('movie_toy_star_3D.mp4', fpath3+'/movie_toy_star_3D.mp4')
    
    for file_name1 in File_Names1:  #Delete the image files
    
        os.remove(fpath2+'/'+file_name1)
        
    for file_name2 in File_Names2:  #Delete the image files
    
        os.remove(fpath3+'/'+file_name2)
        
    return
        
        
        
        
