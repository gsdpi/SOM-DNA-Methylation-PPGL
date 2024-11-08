# numpy + matplotlib + pandas
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist 
import time

from numba import jit

def inisom(x,dims):
	'''
	inisom(x,dims)
	Initialization of the SOM parameters: 
 	S grid nodes "gi" on a 2D lattice
  	and S prototypes/codebooks "mi" of the input data samples

	Parameters
		x: input data (Q,R)
	 dims: list (ni,nj), with the number or rows  and columns 

	Returns
		gi: grid nodes (S,2)
		mi: codebooks (S,R)
	'''
	if len(dims)==2:
		ii,jj = np.meshgrid(np.arange(dims[0]),np.arange(dims[1]))
		gi = np.column_stack((ii.ravel(),jj.ravel()))
		mi = np.random.randn(np.prod(dims),x.shape[1])
	elif len(dims)==3:
		ii,jj,kk = np.meshgrid(np.arange(dims[0]),np.arange(dims[1]),np.arange(dims[2]))
		gi = np.column_stack((ii.ravel(),jj.ravel(),kk.ravel()))
		mi = np.random.randn(np.prod(dims),x.shape[1])
     
	return gi, mi

def fproj(x, gi, mi, dither=False):
	'''
	
	[c, q, y, mse, xest] = fproj(x, gi, mi, dither=False)
	
	Projects the samples of a batch on the SOM
	
 	Parameters
	   x: input data batch (Q,R)
	  gi: grid nodes (S,2)
	  mi: codebooks (S,R)
	  dither: if True, add uniform [-0.5,0.5] to the projections around their node
 
	Returns
	   c: sequence of bmu's (Q,)
	   q: sequence of quantization errors (Q,)
       y: sequence of grid positoins (Q,2)
     mse: mean squared error of the whole batch
	xest: reconstruction of x using best prototypes (Q,R)
	'''
	
	dij  = cdist(mi,x)**2			# distance to all prototypes (S,Q)
	c    = dij.argmin(axis=0)		# best matching unit (Q,)
	q    = dij.min(axis=0)			# quantization error (Q,)
	y    = gi[c,:]					# output trajectory (Q,L)
	mse  = q.mean()					# mean squared error ()
	xest = mi[c,:]					# reconstruction of x (Q,R)

	if dither:
		y = y.astype(float) + np.random.rand(*y.shape) - 0.5

	return c, q, y, mse, xest


def somdist(mi,gi):
	'''
	dist = somdist(mi,gi)

	interneuron distance matrix

	Parameters
	mi: codebooks (S,R)
	gi: grid nodes (S,2)

	Returns
	dist: mean distances from mi to their 1-neighbors (S**2,)

	'''
	S,R = mi.shape	
	dist = np.zeros(S)
	for i in range(S):
		for j in range(S):
			dij = abs(gi[i,0]-gi[j,0]) + abs(gi[i,1]-gi[j,1])
			if dij == 1:
				# if j is neighbor of i, compute |mi - mj|^2
				for r in range(R):
					dist[i] += (mi[i,r]-mi[j,r])*(mi[i,r]-mi[j,r])
	return dist


def somdist_squared_som(mi,ni):
	'''
	dist = somdist_squared_som(mi,ni)

	fast computation of interneuron distance matrix
	only for squared SOM of size (ni,ni)

	Parameters
	mi: codebooks (S,R)
	ni: number of grid rows/cols (grid must be squared, i.e. same number of rows and cols )

	Returns
	dist: mean distances from mi to their 1-neighbors (S**2,)

	'''

	S,R = mi.shape	
 
	if S != ni**2:
		print('''
        somdist_squared_som(mi,ni) is a fast implementation 
        only working for squared grids of size (ni,ni)')
		... use somdist(mi,gi) for for arbitrary nodes gi ''')
		return
 
	dist = np.zeros(S)
	for i in range(S):
		row = i // ni
		col = i % ni
		if row < ni-1:
			j = i + ni
			for r in range(R):
				dist[i] += (mi[i,r]-mi[j,r])*(mi[i,r]-mi[j,r])
		if row > 0:
			j = i - ni
			for r in range(R):
				dist[i] += (mi[i,r]-mi[j,r])*(mi[i,r]-mi[j,r])
		if col < ni-1:
			j = i + 1
			for r in range(R):
				dist[i] += (mi[i,r]-mi[j,r])*(mi[i,r]-mi[j,r])
		if col > 0:
			j = i - 1
			for r in range(R):
				dist[i] += (mi[i,r]-mi[j,r])*(mi[i,r]-mi[j,r])
	return dist




def somdist_rect_som(mi,ni,nj):
	'''
	dist = somdist_rect_som(mi,ni,nj)

	fast computation of interneuron distance matrix
	only for rectangular SOM of size (ni,nj)

	Parameters
	mi: codebooks (S,R)
	ni: number of grid rows (int)
	nj: number of grid cols (int)

	Returns
	dist: mean distances from mi to their 1-neighbors (S,)

	'''

	S,R = mi.shape	
 
	if S != ni*nj:
		print('ERROR: number of codebooks not equal to ni*nj')
		return
 
	dist = np.zeros(S)
	for i in range(S):
		row = i // ni
		col = i % nj
		if row < ni-1:
			j = i + ni
			for r in range(R):
				dist[i] += (mi[i,r]-mi[j,r])*(mi[i,r]-mi[j,r])
		if row > 0:
			j = i - ni
			for r in range(R):
				dist[i] += (mi[i,r]-mi[j,r])*(mi[i,r]-mi[j,r])
		if col < nj-1:
			j = i + 1
			for r in range(R):
				dist[i] += (mi[i,r]-mi[j,r])*(mi[i,r]-mi[j,r])
		if col > 0:
			j = i - 1
			for r in range(R):
				dist[i] += (mi[i,r]-mi[j,r])*(mi[i,r]-mi[j,r])
	return dist



	
def bsom(x,gi,mi,neigh,mask=None):
	'''
	batch SOM algorithm
	
		bsom(x, gi, mi, neigh, mask=None)
	
	Update the SOM prototypes mi for 1 epoch
	
	Parameters
	         x: input data batch (Q,R)
 	        gi: 2D grid nodes (S,2)
	        mi: codebooks/prototypes (S,R)
	     neigh: neighborhood ()
	      mask: list with a subset of variables to be considered for similarity
  
	Returns
	        mi: the updated codebooks
           mse: the mean squared error for the data batch
 	'''


	# distance to all prototypes mi (S,Q)
	if mask:
		# consider only dimensions specified in the mask
		mask = np.array(mask).reshape(1,x.shape[1])
		dij = cdist(mi*mask,x*mask)
	else:
		# compute the distance using all dimensions
		dij = cdist(mi,x)**2
	
 	# best matching unit (Q,)
	c   = dij.argmin(axis=0)
	
 	# neighborhood function (S,Q)
	hci = np.exp(-cdist(gi,gi[c,:],metric='cityblock')/neigh)
	
 	# prototypes are the mean of the input data weighted by neighborhood to the winner
	mi  = np.einsum('qr,sq->sr',x,hci/hci.sum(axis=1,keepdims=True))
	
	mse = np.mean(dij.min(axis=0))
	
	return mi,mse





def csom(x, gi, mi, neigh, mu=0.01, mask=None):
	'''
	online SOM algorithm

		csom(x, gi, mi, neigh, mu=0.01, mask=None)

	Update the SOM prototypes mi for 1 epoch
	
	Parameters
	         x: input data batch (Q,R)
 	        gi: 2D grid nodes (S,2)
	        mi: codebooks/prototypes (S,R)
	     neigh: neighborhood ()
	        mu: learning rate ()
	      mask: list with a subset of variables to be considered for similarity
	      dist: 'L2' (default) or 'L1'
	neigh_type: 'bubble' (default) or 'gaussian'
  
	Returns
	        mi: the updated codebooks
           mse: the mean squared error for the data batch
 	'''
	 	
	# distance from input data samples to codebooks (Q,S)
	
	# if mask, use only a subset of the attributes
	if mask: 
		mask = np.array(mask).reshape(1,x.shape[1])
		dki  = cdist(x*mask,mi*mask)**2
	else:
		dki  = cdist(x,mi)**2

	# bmu for each sample (Q,)
	c = dki.argmin(axis=1)

	# neigborhood from the bmu of each sample c(k) to codebooks (Q,S) 
	hci = np.exp(-cdist(gi[c,:],gi,metric='cityblock')/neigh)

	# tensor with differences between inputs and codebooks (Q,S,R)
	Dkir = x[:,np.newaxis,:]-mi[np.newaxis,:,:]

	# update equation
	mi = mi + mu*np.einsum('ki,kir->ir',hci,Dkir)

	# mean squared error
	mse = np.mean(dki.min(axis=1))   

	return mi,mse