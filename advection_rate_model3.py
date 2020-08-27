'''Calculate the rate of porewater advection that best explains observed [Li] in sediment porewater. Change everything with '***' for another solute. '''

# imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math


'''--------------------------------------------------------------------------'''
# Get data
pw_file = 'test_data.xls'
porosity_file = 'test_data.xls'
solute = 'Li (µmol/L)' #***

myplace = 'raec'

# input directory
in_dir = '../' + myplace + '_data/'
in_pw = in_dir + pw_file
in_porosity = in_dir + porosity_file

# output directory
out_dir = '../' + myplace + '_output/' + 'advection_rate/'

pw_df_all = pd.read_excel(in_pw, sheet_name='pw', index_col=[0,1,2]) # read-in xls file. column headings are taken from row 0
pw_df_all = pw_df_all.apply(pd.to_numeric, args=('coerce',)) # turn all numbers into floats, all strings into NaNs.

print('Which core would you like to model?\nEnter exactly a core name from the list below (no quotation marks):\n')
print(pw_df_all.index.unique(level=0))
core = input('> ')
pw_df_1core = pw_df_all.loc[core]
site = pw_df_1core.index[0][1]

c_obs_all = np.array(pw_df_1core[solute]) # measured concentrations of selected solute
z_obs_all = np.array(pw_df_1core['Depth']) # depths at which measurements were made

por_df_all = pd.read_excel(in_porosity, sheet_name='porosity', index_col=[0,1,2]) # read-in xls file. column headings are taken from row 0

print('Which core would you like to use for porosity data?\nEnter exactly a core name from the list below (no quotation marks):\n')
print(por_df_all.index.unique(level=0))
por_core = input('> ')
por_df_1core = por_df_all.loc[por_core]
por_site = por_df_1core.index[0][1]

por_arr = np.array(por_df_1core['Porosity']) # measured porosity values
por_arr = np.append(por_arr, por_arr[-1]) # duplicate the last measured value
por_z = np.array(por_df_1core['Depth']) # depths at which measurements were made
por_z = np.append(por_z, por_z[-1]+999) #create an atrificial depth value after the final measurement
# may need to increase the added value such that por_z_max > z_max

'''--------------------------------------------------------------------------'''
# Cut out mixed layer
plt.close('all')
plt.plot(c_obs_all, z_obs_all, marker = 'o')
plt.gca().invert_yaxis()
plt.title('Observed ' + solute + ' vs Depth')
plt.show()

print('How many nodes make up a vertical mixed layer? (input an integer value)')
ml_input = int((input('> ')))
ml_depth = z_obs_all[ml_input]

print(f'The selected mixed layer depth is: {ml_depth} cmbsf')

c_obs_cut = c_obs_all[ml_input:]
z_obs_cut = z_obs_all[ml_input:] - ml_depth

'''-------------------------------------------------------------------------------'''
# constants
Dm = 5.34e-10 # molecular diffusion coefficient of Li (m^2/s) ***
z_max = z_obs_cut[-1] # maximum depth (cm)
n = 1099
dz = z_max/n  # step size
z = np.linspace(0, z_max, n+1) # array of z values
D = 2 * (dz**2) # from matrix math. ********fixed***********

r = np.zeros((n+1, 1)) # empty array for reaction rate
for i in range (n+1):
    r[i] = 0#(-1.5e-5)*math.exp(.01*z[i]) # equation that defines reaction rate

'''----------------------------------------------------------------------------'''
#functions

def find_Ds(node):
    '''function to determine sediment diffusion coefficient at a given depth'''
    # depth = ((node/n) * z_max) + ml_depth
    # i = 0
    # while por_z[i] < depth: # find the i closest to depth we are looking at
    #     i += 1
    # por = ((depth - por_z[i-1]) / (por_z[i] - por_z[i-1]) * (por_arr[i] - por_arr[i-1])) + por_arr[i-1] #use that i to estimate porosity at that depth
    # tort = 1 - np.log(por**2) # calculate tortuosity from porosity
    # Ds = (Dm / tort) * (3.154e11) # calculate sediment diffusion coefficient (cm^2/yr)
    # return Ds

# if you take out the above content and just use a constant Ds, the model works.
    por = np.mean(por_arr[:-1]) #average porosity (excluding final artificial value)
    tort = 1 - np.log(por**2) # calculate tortuosity from porosity (Boudreau's law)
    Ds = (Dm / tort) * (3.154e11)
    return Ds #return a constant Ds value based on average porosity
'''----------------------------------------------------------------------------'''
def find_c(w):
    '''a function that approximates the concentration with depth for a given w'''
    s = w / (2 * dz) # from matrix math

# create matrix A
    A = np.zeros((n+1, n+1)) #square matrix, size defined by number of spatial nodes

    for i in range(n+1):
        A[i,i] = (find_Ds(i+1) + 2*find_Ds(i) + find_Ds(i-1)) / D # center diagonal

    for j in range(n):
# goes to n because there's always one less than on the center diagonal
        A[j+1, j] = -(s + ((find_Ds(j+1) + find_Ds(j))/D)) # below center diagonal
        A[j, j+1] = s - ((find_Ds(j+1) + find_Ds(j))/D) # above center diagonal

    #print(A)

# create matrix B
    g1 = c_obs_cut[0] # [Li] at upper boundary
    g2 = c_obs_cut[-1] # [Li] at bottom boundary

    B = np.zeros((n+1, 1)) # matrix of 0s that's n+1 down, 1 across
    for i in range (n+1):
        B[i] = r[i]

    B[0,0] = ((s + ((find_Ds(-1) + find_Ds(0))/ D)) * g1) + r[0] # overriding first value
    B[n,0] = (-(s - ((find_Ds(n-1) + find_Ds(n))/ D)) * g2) +r[n] # overriding the last value

    #print(B)

    c = np.linalg.inv(A) @ B

    return c # returns an array of approximated concentrations (length n)

'''-------------------------------------------------------------------------------'''

def find_error(y_appx, y_ex):
    '''a function to calculate the error: absolute value of the difference between the numerical and observed values'''
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
        #if w == 0:         # why did we need this????????????????????
        #    continue
        c = find_c(w)
        c_arr.append(c)
        c_error = []
        for j in range(len(z_obs_cut)):
            c_error.append(c[int(z_obs_cut[j] * n / z_max),0])
        w_arr.append(w)
        err = find_error(c_error, c_obs_cut)
        err_arr.append(err)
        print(i,round(w,2),round(err,2))
        #print(c_arr)
    k = err_arr.index(min(err_arr))
    print(f"\n-------------\nbest w: {w_arr[k]:.3} cm/yr, error at w = {w_arr[k]:.3}: {err_arr[k]:.3}")
    return w_arr[k], c_arr[k], w_arr, c_arr

'''-------------------------------------------------------------------------------'''
def analytic_sln(w):
    c_analytic = []
    por_analytic = np.mean(por_arr)
    tort_analytic = 1 - np.log(por_analytic**2) # calculate tortuosity from porosity (Boudreau's law)
    Ds_analytic = (Dm / tort_analytic) * (3.154e11)
    c0 = c_obs_cut[0] # [Li] in SW (um)
    cmax = c_obs_cut[-1] # [Li] at bottom boundary

    for i in range(n+1):
        if w == 0:
            w_sub = 1e-10
            C = c0 + (cmax - c0) * ((1 - math.exp((w_sub*z[i])/Ds_analytic))/(1 - math.exp((w_sub*z_max)/Ds_analytic)))
        else:
            C = c0 + (cmax - c0) * ((1 - math.exp((w*z[i])/Ds_analytic))/(1 - math.exp((w*z_max)/Ds_analytic)))
        c_analytic.append(C)
    return c_analytic

'''-------------------------------------------------------------------------------'''
def plot_c(c, w):
    #plot c vs depth
    plt.close('all')
    fs = 14
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot() # a function to plot the numerical solution vs observed values
    ax.plot(c_obs_all, z_obs_all, marker = 'o', label='Observed values')
    ax.plot(c, z + ml_depth, label=f'Model at w = {w:.3} cm/yr')
    ax.plot(c_analytic, z + ml_depth, 'k', label=f'Analytical solution at w = {w:.3} cm/yr')
    plt.gca().invert_yaxis()
    ax.set_title(f'Modeled {solute}, ({core}, Site {site})', fontsize = fs*1.2)
    #ax.set_title('Measured [Li] with depth', fontsize = fs)
    plt.legend()
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.spines['right'].set_visible(False) # don't show the right and bottom axes of each plot
    ax.spines['bottom'].set_visible(False)
    ax.set_xlabel(solute, fontsize = fs)
    ax.set_ylabel('Depth (cmbsf)', fontsize = fs)
    fig.savefig(out_dir + core + '_best_w.png')

    #plot porosity vs depth as 2nd figure
    fig2 = plt.figure(figsize=(6,6))
    ax2 = fig2.add_subplot()
    ax2.plot(por_arr[:-1], por_z[:-1], marker='o')
    ax2.set_title(f'Porosity vs Depth ({por_core}, Site {por_site})', fontsize = fs*1.2)
    ax2.xaxis.tick_top()
    ax2.xaxis.set_label_position('top')
    ax2.spines['right'].set_visible(False) # don't show the right and bottom axes of each plot
    ax2.spines['bottom'].set_visible(False)
    ax2.set_xlabel(r'Porosity', fontsize = fs)
    ax2.set_ylabel('Depth (cmbsf)', fontsize = fs)
    ax2.set_xlim(0, 1)
    plt.gca().invert_yaxis()
    plt.show()
    fig2.savefig(out_dir + por_core + '_porosity.png')


    #plot modeled profiles for each w
    fig3 = plt.figure(figsize=(6,6))
    ax3 = fig3.add_subplot()
    ax3.plot(c_obs_all, z_obs_all, marker = 'o', label='Observed values')
    ax3.plot(c_analytic, z + ml_depth, 'k', label=f'Analytical solution at w = {w:.3} cm/yr')
    for i in range(len(w_arr)):
       ax3.plot(c_arr[i], z+ml_depth, label=f'Model at w= {w_arr[i]:.3} cm/yr')
    ax3.set_title(f'Modeled {solute}, ({core}, Site {site})', fontsize = fs*1.2)
    #ax.set_title('Measured [Li] with depth', fontsize = fs)
    plt.legend()
    ax3.xaxis.tick_top()
    ax3.xaxis.set_label_position('top')
    ax3.spines['right'].set_visible(False) # don't show the right and bottom axes of each plot
    ax3.spines['bottom'].set_visible(False)
    ax3.set_xlabel(solute, fontsize = fs)
    ax3.set_ylabel('Depth (cmbsf)', fontsize = fs)
    plt.gca().invert_yaxis()
    plt.show()
    fig3.savefig(out_dir + core + '_all_w.png')



# Ds_arr = [] # if you want Ds as a function of depth
# for i in range(n+1):
#     Ds = find_Ds(i)
#     Ds_arr.append(Ds)

# execute code. find the best w (w with least error) between -0.3 and -.25. print and plot findings
print("Enter a range of rates you'd like to test in cm/yr\n(- values for upward advection, + values for downward advection):\n")
w_low_input = float(input('\tLower bound: '))
w_high_input = float(input('\n\tUpper bound: '))
print('\n')
best_w, best_c, w_arr, c_arr = find_best_w(w_low_input, w_high_input, 10)
c_analytic = analytic_sln(best_w)
plot_c(best_c, best_w)


#for i in range(len(w_arr)):
 #  plt.plot(c_arr[i], z)
