from pylab import *
from prosail_fortran import run_prosail
import numpy as np
from GaussianProcess import GaussianProcess

def xify(p,params = ['n','xcab','xcar','cbrown','xcw','xcm','xala','bsoil','psoil','hspot','xlai']):
  '''
  Fix the state key names by putting x on where appropriate
  '''
  retval = {}
  for i in p.keys():
    if 'x' + i in params:
      retval['x' + i] = p[i]
    else:
      retval[i] = p[i]
  return retval

def invtransform(params,logvals = {'ala': 90., 'lai':-2.0, 'cab':-100., 'car':-100., 'cw':-1/50., 'cm':-1./100}):
  retval = {}
  for i in xify(params).keys():
    if i[0] != 'x':
      retval[i] = params[i]
    else:
      rest = i[1:]
      try:
        if logvals[rest] < 0:
          retval[rest] = logvals[rest] * np.log(params[i])
        else:
          retval[rest] = params[i] * logvals[rest]
      except:
        retval[rest] = params[i]
  return retval

def transform(params,logvals = {'ala': 90., 'lai':-2.0, 'cab':-100., 'car':-100., 'cw':-1/50., 'cm':-1./100}):
  retval = {}
  for i in xify(params).keys():
    if i[0] != 'x':
      retval[i] = params[i]
    else:
      rest = i[1:]
      try:
        if logvals[rest] < 0:
          retval[i] = np.exp(params[rest] / logvals[rest])
        else:
          retval[i] = params[rest] / logvals[rest]
      except:
        retval[i] = params[i]
  return retval

def limits():
  '''
  Set limits for parameters
  '''
  params = ['n','xcab','xcar','cbrown','xcw','xcm','xala','bsoil','psoil','hspot','xlai']
  pmin = {}
  pmax = {}
  # set up defaults
  for i in params:
    pmin[i] = 0.001
    pmax[i] = 1. - 0.001
  # now specific limits
  # These from Feret et al.
  pmin['xlai'] = transform({'lai':15.})['xlai']
  pmin['xcab'] = transform({'cab':0.2})['xcab']
  pmax['xcab'] = transform({'cab':76.8})['xcab']
  pmin['xcar'] = transform({'car':25.3})['xcar']
  pmax['xcar'] = transform({'car':0.})['xcar']
  pmin['cbrown'] = 1.
  pmax['cbrown'] = 0.
  pmin['xcm'] = transform({'cm':0.0017})['xcm']
  pmax['xcm'] = transform({'cm':0.0331})['xcm']
  pmin['xcw'] = transform({'cw':0.0043})['xcw']
  pmax['xcw'] = transform({'cw':0.0713})['xcw']

  pmin['n'] = 0.8
  pmax['n'] = 2.5
  pmin['bsoil'] = 0.0
  pmax['bsoil'] = 2.0
  pmin['psoil'] = 0.0
  pmax['psoil'] = 1.0
  pmin['xala'] = transform({'ala':90.})['xala']
  pmax['xala'] = transform({'ala':0.})['xala']
  for i in params:
    if pmin[i] > pmax[i]:
      tmp = pmin[i]
      pmin[i] = pmax[i]
      pmax[i] = tmp
  return (pmin,pmax)


def fixnan(x):
  '''
  the RT model sometimes fails so we interpolate over nan

  This method replaces the nans in the vector x by their interpolated values
  '''
  for i in xrange(x.shape[0]):
    sample = x[i]
    ww = np.where(np.isnan(sample))
    nww = np.where(~np.isnan(sample))
    sample[ww] = np.interp(ww[0],nww[0],sample[nww])
    x[i] = sample
  return x

def addDict(input,new):
  '''
  add dictionary items
  '''
  for i in new.keys():
    try:
      input[i] = np.hstack((np.atleast_1d(input[i]),np.atleast_1d(new[i])))
    except:
      input[i] = np.array([new[i]]).flatten()
  return input

def unpack(params):
  '''
  Input a dictionary and output keys and array
  '''
  inputs = []
  keys = np.sort(params.keys())
  for i,k in enumerate(keys):
    inputs.append(params[k])
  inputs=np.array(inputs).T
  return inputs,keys

def pack(inputs,keys):
  '''
  Input keys and array and output dict
  '''
  params = {}
  for i,k in enumerate(keys):
    params[k] = inputs[i]
  return params

def sampler(pmin,pmax):
  '''
  Random sample
  '''
  params = {}
  for i in pmin.keys():
    params.update(invtransform({i:pmin[i] + np.random.rand()*(pmax[i]-pmin[i])}))
  return params


def samples(n=2):
  '''
  Random samples over parameter space
  '''
  (pmin,pmax) = limits()
  s = {}
  for i in xrange(n):
    s = addDict(s,sampler(pmin,pmax))
  return s


def reconstruct(gps,SB,inputs):
  '''
  spectral reconstruction from GPs and spectral basis functions
  '''
  fwd = 0
  inputs = np.atleast_2d(np.array(inputs))
  for i in xrange(SB.shape[0]):
    pred_mu, pred_var, grad  = gps[i].predict(inputs)
    fwd += np.matrix(pred_mu).T * np.matrix(SB[i])
  return np.array(fwd)

def reconstructD(gps,SB,inputs):
  '''
  spectral reconstruction from GPs and spectral basis functions
  '''
  fwd = 0
  inputs = np.atleast_2d(np.array(inputs))
  deriv = np.zeros((inputs.shape[1],SB[0].shape[0]))
  derivt = np.zeros((inputs.shape[1],SB[0].shape[0]))

  for i in xrange(SB.shape[0]):
    pred_mu, pred_var, grad  = gps[i].predict(inputs)
    fwd += np.matrix(pred_mu).T * np.matrix(SB[i])
    deriv += np.matrix(grad).T * np.matrix(SB[i])
  # check code , but its the same
  #for j in xrange(inputs.shape[1]):
  #  for i in xrange(SB.shape[0]):
  #    pred_mu, pred_var, grad  = gps[i].predict(inputs)
  #    derivt[j,:] += grad[:,j] * SB[i]
  return np.array(fwd)[0],deriv


def brdfModel(s,vza,sza,raa):
  '''
  Run the full BRDF model for parameters in s
  '''
  
  try:
    brdf = []
    for i in xrange(len(s['n'])):
      brdf.append(run_prosail(s['n'][i],s['cab'][i],s['car'][i],s['cbrown'][i],s['cw'][i],s['cm'][i],\
                s['lai'][i],s['ala'][i],0.0,s['bsoil'][i],s['psoil'][i],s['hspot'][i],\
                vza[i],sza[i],raa[i],2))
  except:
    brdf = []
    for i in xrange(len(s['n'])):
      brdf.append(run_prosail(s['n'][i],s['cab'][i],s['car'][i],s['cbrown'][i],s['cw'][i],s['cm'][i],\
                s['lai'][i],s['ala'][i],0.0,s['bsoil'][i],s['psoil'][i],s['hspot'][i],\
                vza,sza,raa,2))
  return (vza,sza,raa),transform(s),fixnan(np.array(brdf))
  

def lut(n=2,vza=0.,sza=45.,raa=0.,brdf=None,bands=None):
  '''
  Generate a random LUT for given viewing and illumination geometry
  '''
  # get the parameter limits
  (pmin,pmax) = limits()

  # generate some random samples over the parameter space
  s = samples(n=n)
  return brdfModel(s,vza,sza,raa)

import pickle
# full sprectum
bands = None
thresh = 0.98
n = 150
lutfile = 'lutTest.dat.npz'
pickleFile = 'lutTest.dat.pkl'

vza = 0.
sza = 45.
raa = 0.

try:
  gps = pickle.load(open(pickleFile,'rb'))
  npzfile = np.load(lutfile)
  angles,train_params,train_brf = npzfile['angles'],npzfile['train_params'],npzfile['train_brf']
  theta_min0,theta_min1 = npzfile['theta_min0'],npzfile['theta_min1']
  s,SB,train_SB = npzfile['s'],npzfile['SB'],npzfile['train_SB']
  readLUT = True
  print 'Read lutfile %s and %s'%(lutfile,pickleFile)
except:
  readLUT = False
  print 'Failed to read lutfile %s'%lutfile
  angles,train_params,train_brf = lut(n,bands=bands,vza=vza,sza=sza,raa=raa)
  # spectral reduction
  U, s, V = np.linalg.svd(train_brf, full_matrices=True)
  wt = s.cumsum()/s.sum()
  ww = np.where(wt<=thresh)
  SB = V[ww[0],:]

  # now project the training data onto these axes
  train_SB = np.dot(train_brf,SB.T).T

  # unpack params
  inputs_t,keys = unpack(train_params)

  # set some arrays to collect information
  gps = ['']*train_SB.shape[0]
  theta_min0 = ['']*train_SB.shape[0]
  theta_min1 = ['']*train_SB.shape[0]

  # loop over each dimenstion and train a gp
  for i in xrange(train_SB.shape[0]):
    yields_t = train_SB[i]
    gps[i] = GaussianProcess ( inputs_t, yields_t )
    theta_min0[i], theta_min1[i] = gps[i].learn_hyperparameters (n_tries=2)

  pickle.dump(gps,open(pickleFile,'wb'))
  np.savez(lutfile,angles=angles,train_params=train_params,\
                  train_brf=train_brf,\
                  theta_min0=theta_min0,theta_min1=theta_min1,\
                  s=s,SB=SB,train_SB=train_SB)

wl = np.arange(400,2501).astype(float)

plt.clf()
for i in xrange(train_SB.shape[0]):
  plt.plot(wl,SB[i],label='PC %d (%.2f)'%(i,(s/s.sum())[i]))

plt.legend()
plt.xlim(wl.min(),wl.max())
plt.show()


# now look at e.g. reflectance as a function of LAI
(pmin,pmax) = limits()

xlais = np.arange(pmin['xlai'],pmax['xlai'],0.1)
ones = np.ones_like(xlais)
params = {}
for i in pmin.keys():
  params.update(invtransform({i:ones*(pmin[i]+pmax[i])*0.5}))
params.update(invtransform({'xlai':xlais}))

# now run true model
brdf = brdfModel(params,vza,sza,raa)[-1]
# now emulate (dont forget to transform!)
inputs,keys = unpack(transform(params))
brdfe = reconstruct(gps,SB,inputs)
plt.clf()
plt.plot(np.array([wl]*10).T,brdf.T,'k-')
plt.plot(np.array([wl]*10).T,brdfe.T,'r--')
plt.xlim(wl[0],wl[-1])
plt.xlabel('wavelength (nm)')
plt.ylabel('reflectance')
plt.title('true (black) and emulated (red) reflectance for varying LAI')
plt.show()


##### We now generate an independent dataset and use that for testing the GP

angles,test_params,test_brf = lut(n,bands=bands)
test_SB = np.dot(test_brf,SB.T).T

# unpack params
inputs_v,keys = unpack(test_params)

plt.clf()
# loop over each dimenstion and test a gp
for i in xrange(train_SB.shape[0]):
  # This is the set of true projected parameters
  yields_v = test_SB[i]
  # This is what we predict from the emulator
  pred_mu, pred_var, deriv  = gps[i].predict ( inputs_v )

  plt.plot(yields_v,pred_mu,'+',label='PC %d'%i)

plt.legend()
plt.show()

# check derivatives
inputs_t,keys = unpack(train_params.tolist())
xf = lambda x:gps[0].predict ( np.atleast_2d(x) )[0]
import scipy.optimize
trueDerive = []
for i in xrange(inputs_t.shape[0]):
  trueDerive.append(scipy.optimize.approx_fprime(inputs_t[i],xf,1e-10))
pred_mu, pred_var, deriv  = gps[0].predict ( np.atleast_2d(inputs_t) )
# plot
plt.plot(np.array(trueDerive).flatten(),deriv.flatten(),'g+')



# Now look at some example reconstructions

fwd = reconstruct(gps,SB,inputs_v)

i = 0
true = test_brf[i]
em = fwd[i]
plt.clf()
plt.plot(wl,true,'k-')
plt.plot(wl,em,'r+')

plt.xlabel('wavelength (nm)')
plt.ylabel('reflectance')
plt.show()



i = 50
true = test_brf[i]
em = fwd[i]
plt.clf()
plt.plot(wl,true,'k-')
plt.plot(wl,em,'r+')

plt.xlabel('wavelength (nm)')
plt.ylabel('reflectance')
plt.show()



i = 100
true = test_brf[i]
em = fwd[i]
plt.clf()
plt.plot(wl,true,'k-')
plt.plot(wl,em,'r+')

plt.xlabel('wavelength (nm)')
plt.ylabel('reflectance')
plt.show()



# We now attempt to use the emulator to solve inverse problems (ultimately to allow fast data assimilation).


# function to do the minimisation in spectral space
def funcd(params,brf,SB,gps):
  fwd,derivs = reconstructD(gps,SB,params)
  obs = brf
  d = (fwd - obs)
  Jprime = np.matrix(d) * np.matrix(derivs).T
  e = np.sum(d*d)
  return 0.5*e,Jprime.T

from  scipy.optimize import fmin_l_bfgs_b

# form the state vector

(pmin,pmax) = limits()
keys = np.sort(pmin.keys())
ppmin = unpack(pmin)[0]
ppmax = unpack(pmax)[0]

b = [(ppmin[i],ppmax[i]) for i in xrange(len(keys))]
x = []
for k in keys:
  x.append((pmin[k]+pmax[k])*0.5)

# function to do the minimisation in EOF space
def funcd2(params,projectedBrf,gps):
  params = np.atleast_2d(np.array(params))
  e = 0.
  Jprime = np.zeros_like(params)
  for i in xrange(len(gps)):
    fwd,sd,deriv = gps[i].predict(params)
    d = (fwd[0] - projectedBrf[i]) # /sd[0]
    Jprime += d*deriv
    e += d*d
  return 0.5*e,Jprime

# project the test_brf data
test_SB = np.dot(test_brf,SB.T)
def solver2(i,noshow=True):
  params = fmin_l_bfgs_b(funcd2,x,bounds=b,args=(test_SB[i],gps),iprint=0)[0]
  fwd = reconstruct(gps,SB,params)[0]
  if not noshow:
    plt.figure(i)
    plt.clf()
    plt.plot(wl,test_brf[i],'k-')
  plt.plot(wl,fwd,'g+')
  plt.show()
  print 'parameters2: ',invtransform(pack(params,keys))
  print 'true       : ',invtransform(pack(inputs_v[i],keys))


def solver(i,noshow=True):
  params = fmin_l_bfgs_b(funcd,x,bounds=b,args=(test_brf[i],SB,gps),iprint=0)[0]
  fwd = reconstruct(gps,SB,params)[0]
  plt.figure(i)
  plt.clf()
  plt.plot(wl,test_brf[i],'k-')
  plt.plot(wl,fwd,'r+')
  if not noshow:
    plt.show()
  print 'parameters1: ',invtransform(pack(params,keys))
  if not noshow:
    print 'true       : ',invtransform(pack(inputs_v[i],keys))

solver(0)
solver2(0)

solver(25)
solver2(25)

solver(50)
solver2(50)

solver(75)
solver2(75)

solver(100)
solver2(100)
