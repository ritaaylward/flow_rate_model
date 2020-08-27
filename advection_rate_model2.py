'''Code to calculate the rate of advection that best explains observed vaules of Li in sediment porewater'''

# imports
import numpy as np
import matplotlib.pyplot as plt
import math

'''-------------------------------------------------------------------------------'''
# constants
Dm = 5.34e-10 # molecular diffusion coefficient (m^2/s)
z_max = 630 # maximum depth (cm)
n = 999
dz = z_max/n  # step size
z = np.linspace(0, z_max, n+1) # array of z values
D = 2 * (dz**2) # ****************** fixed *****************  was D = 1 / (2 * (dz**2))

r = np.zeros((n+1, 1)) # empty array for reaction rate
for i in range (n+1):
    r[i] = 0#(-1.5e-5)*math.exp(.01*z[i]) # equation that defines reaction rate

'''----------------------------------------------------------------------------'''

# example core data that we're trying to model
c_jpc1 = [29.3, 25.7, 24.0, 21.4, 19.4, 17.0, 16.0, 15.1, 14.8, 15.1, 15.3] # measured concentrations of Li
z_jpc1 = [0.0, 55.0, 100.0, 155.0, 205.0, 275.0, 355.0, 425.0, 505.0, 565.0, 630.0] # depths at which measurements were made

por_arr = np.array([[85, 0.647610811], [245, 0.579964818], [300, 0.564415649], [350, 0.548748955], [420, 0.495946785], [500, 0.454188439], [570, 0.495235996], [710, 0.486855265], [720, 0.258169139], [775, 0.449866041], [9999, 0.449866041]]) # measured porosity values (first element = depth, second = porosity). Artificial value added at the end (depth 9999)

'''----------------------------------------------------------------------------'''

def find_Ds(node): # function to determine sediment diffusion coefficient at a given depth
    # depth = ((node/n) * z_max) + 145
    # i = 0
    # while por_arr[i][0] < depth:
    #     i += 1
    # por = ((depth - por_arr[i-1][0]) / (por_arr[i][0] - por_arr[i-1][0]) * (por_arr[i][1] - por_arr[i-1][1])) + por_arr[i-1][1]
    # tort = 1 - np.log(por**2) # tortuosity
    # Ds = (Dm / tort) * (3.154e11) # sediment diffusion coefficient (cm^2/yr)
    # return Ds

# if you take out the above content and just use a constant Ds, the model works.
    por = np.mean(por_arr[:-1,1]) # average porosity (excluding the last artificial value)
    tort = 1 - np.log(por**2) # calculate tortuosity from porosity (Boudreau's law)
    Ds = (Dm / tort) * (3.154e11)
    return Ds #return a constant Ds value based on average porosity
'''----------------------------------------------------------------------------'''

def find_c(w): # a function that approximates the concentration with depth for a given w
    s = w / (2 * dz)
    #p = r / w
# create matrix A
    A = np.zeros((n+1, n+1)) #square matrix, size defined by number of spatial nodes

    for i in range(n+1):
        A[i,i] = (find_Ds(i+1) + 2*find_Ds(i) + find_Ds(i-1)) / D # center diagonal

    for j in range(n):
# goes to n because there's always one less than on the center diagonal
        A[j+1, j] = -(s + ((find_Ds(j-1) + find_Ds(j))/D)) # below center diagonal

        A[j, j+1] = s - ((find_Ds(j+1) + find_Ds(j))/D) # above center diagonal

    #print(A)

# create matrix B
    g1 = c_jpc1[0] # [Li] at upper boundary
    g2 = c_jpc1[-1] # [Li] at bottom boundary

    B = np.zeros((n+1, 1)) # matrix of 0s that's n+1 down, 1 across
    for i in range (n+1):
        B[i] = r[i]

    B[0,0] = ((s + ((find_Ds(-1) + find_Ds(0))/ D)) * g1) + r[0] # overriding first value
    B[n,0] = (-(s - ((find_Ds(n-1) + find_Ds(n))/ D)) * g2) +r[n] # overriding the last value

    #print(B)

    c = np.linalg.inv(A) @ B

    return c # returns an array of approximated concentrations (length n)

'''-------------------------------------------------------------------------------'''

#print(c)
def find_error(y_appx, y_ex): # a function to calculate the error: absolute value of the difference between the numerical and observed values
    y_error = sum(abs(np.subtract(y_appx, y_ex)))
    return y_error

'''-------------------------------------------------------------------------------'''

def find_best_w(w_low, w_high, w_nodes):
    err_arr = []
    w_arr = []
    c_arr = []
    print("i   w   error\n-------------") # labels for iteration prints
    for i in range(w_nodes+1):
        w = w_low + ((w_high-w_low)*(i/w_nodes))
        if w == 0:
            continue
        c = find_c(w)
        c_arr.append(c)
        c_error = []
        for j in range(len(z_jpc1)):
            c_error.append(c[int(z_jpc1[j] * n / z_max),0])
        w_arr.append(w)
        err = find_error(c_error, c_jpc1)
        err_arr.append(err)
        print(i,round(w,2),round(err,2))
        #print(c_arr)
    k = err_arr.index(min(err_arr))
    print(f"best w: {w_arr[k]:.3} cm/yr, error at w = {w_arr[k]:.3}: {err_arr[k]:.3}")
    return w_arr[k], c_arr[k], w_arr, c_arr

'''-------------------------------------------------------------------------------'''
def analytic_sln(w): # function to find the analytical solution
    c_analytic = []
    por_analytic = np.mean(por_arr[:,1])
    tort_analytic = 1 - np.log(por_analytic**2) # calculate tortuosity from porosity (Boudreau's law)
    Ds_analytic = (Dm / tort_analytic) * (3.154e11)
    c0 = c_jpc1[0] # [Li] at upper boundary
    cmax = c_jpc1[-1] # [Li] at bottom boundary

    for i in range(n+1):
        if w == 0: # code breaks if w = 0, so substitute a w that's almost 0
            w_sub = 1e-10
            C = c0 + (cmax - c0) * ((1 - math.exp((w_sub*z[i])/Ds_analytic))/(1 - math.exp((w_sub*z_max)/Ds_analytic)))
        else:
            C = c0 + (cmax - c0) * ((1 - math.exp((w*z[i])/Ds_analytic))/(1 - math.exp((w*z_max)/Ds_analytic)))
        c_analytic.append(C)
    return c_analytic
'''-------------------------------------------------------------------------------'''
def plot_c(c, w): # function to plot results
    plt.close('all')
    fs = 14 #font size

    fig = plt.figure(figsize=(6,6)) #plot the concentration profile assuming the "best" advection rate
    ax = fig.add_subplot()
    ax.plot(c_jpc1, z_jpc1, marker = 'o', label='Observed values')
    ax.plot(c,z, label=f'Model at w = {w:.3} cm/yr')
    ax.plot(c_analytic, z, 'k', label=f'Analytical solution at w = {w:.3} cm/yr')
    plt.gca().invert_yaxis()
    ax.set_title('Modeled [Li] Given Advection Rate w (cm/yr)', fontsize = fs*1.2)
    #ax.set_title('Measured [Li] with depth', fontsize = fs)
    plt.legend()
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.spines['right'].set_visible(False) # don't show the right and bottom axes of each plot
    ax.spines['bottom'].set_visible(False)
    ax.set_xlabel(r'[Li] ($\mu$m)', fontsize = fs)
    ax.set_ylabel('Depth (cmbsf)', fontsize = fs)
    plt.show()

    fig3 = plt.figure(figsize=(6,6)) #plot the profiles for all advection rates tested
    ax3 = fig3.add_subplot()
    ax3.plot(c_jpc1, z_jpc1, marker = 'o', label='Observed values')
    ax3.plot(c_analytic, z, 'k', label=f'Analytical solution at w = {w:.3} cm/yr')
    for i in range(len(w_arr)):
       ax3.plot(c_arr[i], z, label=f'Model at w= {w_arr[i]:.3} cm/yr')
    ax3.set_title(f'Modeled [Li] For all Advection Rates w (cm/yr)', fontsize = fs*1.2)
    #ax.set_title('Measured [Li] with depth', fontsize = fs)
    plt.legend()
    ax3.xaxis.tick_top()
    ax3.xaxis.set_label_position('top')
    ax3.spines['right'].set_visible(False) # don't show the right and bottom axes of each plot
    ax3.spines['bottom'].set_visible(False)
    ax3.set_xlabel(r"[Li] ($\mu$m)", fontsize = fs)
    ax3.set_ylabel('Depth (cmbsf)', fontsize = fs)
    plt.gca().invert_yaxis()
    plt.show()

# execute code. find the best w (w with least error) within a specified range. print and plot findings
best_w, best_c, w_arr, c_arr = find_best_w(-0.5,-0.2,10)
c_analytic = analytic_sln(best_w)
plot_c(best_c, best_w)


#for i in range(len(w_arr)):
#    plt.plot(c_arr[i], z)
