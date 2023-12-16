import os
import time
import numpy as np
import pandas as pd
from numpy.linalg import inv

def fun(x, pis):
    return np.min([np.array(x) * pis, np.ones(len(pis))], axis=0).sum()

def find_position(nums, target):
    # Use bisection to find the position of target
    left = 0
    right = len(nums) - 1
    mid = -1
    while left <= right:
        mid = left + (right - left) // 2
        temp = fun(1/nums[mid], nums)
        if temp < target:
            left = mid + 1
        elif temp > target:
            right = mid - 1
        else:
            return mid
    if fun(1/nums[mid], nums) < target:
        return mid + 1
    else:
        return mid

def find_c(pis, r):
    pis_temp = pis.copy()  
    pis_temp.sort()
    pis_temp = np.clip(pis_temp, 1e-10, None) # Modification to zero probabilities.
    n = len(pis_temp)  # Note that the expected subsample size is len(pis)*r.
    c0 = (n * r) / sum(pis_temp)
    if c0 * pis_temp[-1] <= 1:
        c = c0
    else:
        pis_inverse = pis_temp.copy()
        pis_inverse = sorted(pis_inverse, reverse=True)
        m_prime = find_position(pis_inverse, n*r)
        m = n - m_prime
        c = (n * r - (n - m)) / sum(pis[:m])
    return c


class DataGeneration(): # Time-invariant harzard
    def __init__(self, hazard):
        self.hazard = hazard
        self.N = len(hazard)

    # True failure time for time-invariant intensity
    def failureTime(self):
        failureTime = np.random.exponential(size=self.N, scale= 1/self.hazard)
        return failureTime

    # Censoring time, controlled by parameter mu
    def censoringTime(self, mu):
        temp = np.random.exponential(size=self.N, scale= 1/mu)
        result = np.clip(temp, None, 6)  # set the length of study as 6
        return result

    # Get observed follow-up time and number of controls
    def timeAndControls(self, failureTime, censoringTime):
        result = np.minimum(failureTime, censoringTime)
        uncensorIndicator = (failureTime <= censoringTime).astype(int)
        censoringRate = uncensorIndicator.mean()

        return result, uncensorIndicator, censoringRate

    def getData(self, mu, notExpFailureTime=None):
        if notExpFailureTime is None:
            a, b, c = self.timeAndControls(self.failureTime(), self.censoringTime(mu))
        else:
            a, b, c = self.timeAndControls(notExpFailureTime, self.censoringTime(mu))
        delta = np.ones(self.N)
        pi = np.ones(self.N)
        result = pd.DataFrame({'time': a, 'event': b, 'subsample_indicator': delta, 'subsample_prob': pi}) # Use a dataframe to represent censored data
        return result


class coxSampling():
    def __init__(self, z, data, beta):
        self.N = data.shape[0]
        self.p = z.shape[1]
        z_DF = pd.DataFrame(z)
        self.z_names = z_DF.columns
        self.z = z_DF.values
        self.data = data
        self.beta = beta
        [self.uft, self.uft_map, self.uft_ix, self.nuft, self.risk_enter] = self.indices()

    def indices(self):
        # All failure times
        ift = np.flatnonzero(self.data.event.values == 1)
        ft = self.data.time.values[ift]

        # Unique failure times
        uft = np.unique(ft)
        nuft = len(uft)

        # Indices of cases that fail at each unique failure time.
        uft_map = dict([(x, i) for i,x in enumerate(uft)])
        uft_ix = [[] for k in range(nuft)] 
        for ix,ti in zip(ift,ft):
            uft_ix[uft_map[ti]].append(ix)

        # Indices of cases (failed or censored) that enter the risk set at each unique failure time.
        risk_enter1 = [[] for k in range(nuft)]
        for i,t in enumerate(self.data.time.values):
            ix = np.searchsorted(uft, t, "right") - 1
            if ix >= 0:
                risk_enter1[ix].append(i)
        risk_enter = [np.asarray(x, dtype=np.int32) for x in risk_enter1]

        return [uft, uft_map, uft_ix, nuft, risk_enter]
    
    def pi(self):
        linpred = np.dot(self.z, self.beta)
        linpred -= linpred.max()
        e_linpred = np.exp(linpred)

        # Initialize the A matrix
        A = np.zeros([self.p, self.p])

        # Initialize the meant matrix
        meat = np.zeros_like(self.z)
        
        # Record the integral (reversely)
        int_dlambda = np.zeros(self.nuft + 1)
        int_z_bar_dlambda = np.zeros((self.nuft + 1, self.p))

        # Record the location of event time of each cases when inserted to the sequence of unique failure times (reversely)
        loc_event_time = np.zeros(self.N)
        
        xp0, xp1 = 0., 0.
        # Iterate backward through the unique failure times.
        for i in range(self.nuft)[::-1]:
            ix = self.risk_enter[i] # Indices of new cases entering the risk set.
            ixf = self.uft_ix[i] # Indices of cases that failed at this time
            loc_event_time[ix] = i + 1 # Note to plus 1 here

            xp0 += e_linpred[ix].sum()
            xp1 += (e_linpred[ix][:,None] * self.z[ix,:]).sum(0)
            z_bar = xp1/xp0
            zNt = self.z[ixf,:] - z_bar

            # Update A matrix
            A += np.einsum('ki,kj->ij', zNt, zNt)

            # Update the jump part of meat matrix
            meat[ixf,:] += zNt

            # Update the integral used to compute the compensator part of meat matrix
            dlambda = len(ixf)/xp0
            int_dlambda[i+1] += dlambda
            int_z_bar_dlambda[i+1,:] += z_bar * dlambda

        # Compute the compensator part of meat matrix
        int_dlambda = int_dlambda.cumsum()
        int_z_bar_dlambda = int_z_bar_dlambda.cumsum(0)
        loc_event_time = loc_event_time.astype(int)
        meat -= self.z * (e_linpred * int_dlambda[loc_event_time])[:,None]
        meat += e_linpred[:,None] * int_z_bar_dlambda[loc_event_time, :]

        A /= self.N
        invA = inv(A)
        trace_of_var = np.einsum('ij,ij->i', meat.dot(invA), meat.dot(invA))
        result = np.sqrt(trace_of_var)
        return result
    
    def getSubsample(self, subsampling_rate):
        temp = self.pi()

        # Optimal subsampling
        c1 = find_c(temp, subsampling_rate)
        pi = np.clip(c1 * np.array(temp), None, 1)
        
        sub_data = self.data.copy()
        result = np.random.binomial(n=1, p=pi, size=self.N)
        sub_data.subsample_prob = pi
        sub_data.subsample_indicator = result

        sub_data = sub_data.loc[sub_data.subsample_indicator == 1]
        sub_z = pd.DataFrame(self.z.copy()[result == 1],
                            index = sub_data.index,
                            columns = self.z_names
                            )
        return sub_z, sub_data


class fullSampleInference():
    def __init__(self, z, data):       
        self.N = data.shape[0]
        self.p = z.shape[1]
        z_DF = pd.DataFrame(z)
        self.z_names = z_DF.columns
        self.z = z_DF.values
        self.data = data
        [self.uft, self.uft_map, self.uft_ix, self.nuft, self.risk_enter] = self.indices()

    def indices(self):
        # All failure times
        ift = np.flatnonzero(self.data.event.values == 1)
        ft = self.data.time.values[ift]

        # Unique failure times
        uft = np.unique(ft)
        nuft = len(uft)

        # Indices of cases that fail at each unique failure time
        uft_map = dict([(x, i) for i,x in enumerate(uft)])
        uft_ix = [[] for k in range(nuft)]
        for ix,ti in zip(ift,ft):
            uft_ix[uft_map[ti]].append(ix)

        # Indices of cases (failed or censored) that enter the risk set at each unique failure time.
        risk_enter1 = [[] for k in range(nuft)]
        for i,t in enumerate(self.data.time.values):
            ix = np.searchsorted(uft, t, "right") - 1
            if ix >= 0:
                risk_enter1[ix].append(i)

        risk_enter = [np.asarray(x, dtype=np.int32)
                                    for x in risk_enter1]
        return [uft, uft_map, uft_ix, nuft, risk_enter]

    def grad_and_hess(self, beta): 
        linpred = np.dot(self.z, beta)        
        linpred -= linpred.max()
        e_linpred = np.exp(linpred)

        xp0, xp1, xp2 = 0., 0., 0.
        like, grad, hess, meat = 0., 0., 0., 0.

        for i in range(self.nuft)[::-1]:
            ix = self.risk_enter[i] # Update for new cases entering the risk set.
            ixf = self.uft_ix[i] # Account for all cases that fail at this point.

            xp0 += e_linpred[ix].sum()
            xp1 += (e_linpred[ix][:,None] * self.z[ix,:]).sum(0)
            xp2 += self.z[ix,:].T.dot(e_linpred[ix][:,None] * self.z[ix,:])
            z_bar = xp1 / xp0
            zNt = self.z[ixf,:] - z_bar
            
            # Update likelihood
            like += (linpred[ixf] - np.log(xp0)).sum()

            # Update gradient
            grad += zNt.sum(0) 

            # Update hessian
            hess +=  (-xp2/xp0 + np.outer(z_bar, z_bar)) * len(ixf)

            # Update meat matrix in the sandwich formula
            meat += zNt.T.dot(zNt)

        return [-like, -grad, -hess, meat]

    def Newton(self, beta0=None, stepsize=1e-2, max_iter=1e3, tol = 1e-1):
        if (beta0 == None).any():
            beta0 = np.zeros(self.p)
        beta = beta0.copy()

        for t in np.arange(max_iter):
            loss, grad, hess, meat = self.grad_and_hess(beta)
            grad_norm = np.linalg.norm(grad, ord=2)
            print('iteration: %d, loss: %.2f, norm of gradient: %.2f' %(t+1, loss, grad_norm))

            # Stop iteration if the norm of gradient is smaller than tol.
            if (grad_norm <= tol):
                break
            
            # Newton update
            beta -= stepsize * inv(hess).dot(grad)

        # The plug-in estimate of standard error
        inv_hess = inv(hess)
        temp = inv_hess.dot(meat).dot(inv_hess)
        se_plugin = np.sqrt(np.diag(temp))

        return [beta, se_plugin, int(t)]
    

class subSampleInference():
    def __init__(self, z, data):       
        self.N = data.shape[0]
        self.p = z.shape[1]
        z_DF = pd.DataFrame(z)
        self.z_names = z_DF.columns
        self.z = z_DF.values
        self.data = data
        [self.uft, self.uft_map, self.uft_ix, self.nuft, self.risk_enter] = self.indices()

    def indices(self):
        # All failure times
        ift = np.flatnonzero(self.data.event.values == 1)
        ft = self.data.time.values[ift]

        # Unique failure times
        uft = np.unique(ft)
        nuft = len(uft)

        # Indices of cases that fail at each unique failure time
        uft_map = dict([(x, i) for i,x in enumerate(uft)])
        uft_ix = [[] for k in range(nuft)]
        for ix,ti in zip(ift,ft):
            uft_ix[uft_map[ti]].append(ix)

        # Indices of cases (failed or censored) that enter the risk set at each unique failure time.
        risk_enter1 = [[] for k in range(nuft)]
        for i,t in enumerate(self.data.time.values):
            ix = np.searchsorted(uft, t, "right") - 1
            if ix >= 0:
                risk_enter1[ix].append(i)

        risk_enter = [np.asarray(x, dtype=np.int32)
                                    for x in risk_enter1]
        return [uft, uft_map, uft_ix, nuft, risk_enter]

    def grad_and_hess(self, beta): 
        linpred = np.dot(self.z, beta) - np.log(self.data.subsample_prob.values) # IPW
        linpred -= linpred.max()
        e_linpred = np.exp(linpred)

        xp0, xp1, xp2 = 0., 0., 0.
        like, grad, hess = 0., 0., 0.

        for i in range(self.nuft)[::-1]:
            ix = self.risk_enter[i] # Update for new cases entering the risk set.
            ixf = self.uft_ix[i] # Account for all cases that fail at this point.

            xp0 += e_linpred[ix].sum()
            xp1 += (e_linpred[ix][:,None] * self.z[ix,:]).sum(0)
            xp2 += self.z[ix,:].T.dot(e_linpred[ix][:,None] * self.z[ix,:])
            z_bar = xp1 / xp0
            zNt = (self.z[ixf,:] - z_bar) / self.data.subsample_prob.values[ixf][:,None] # IPW
            
            # Update likelihood
            like += (linpred[ixf] - np.log(xp0)).sum()

            # Update gradient
            grad += zNt.sum(0)

            # Update hessian
            hess +=  (-xp2/xp0 + np.outer(z_bar, z_bar)) * (1/self.data.subsample_prob.values[ixf]).sum()

        return [-like, -grad, -hess]
    
    def se_est(self, beta):
        linpred = np.dot(self.z, beta) - np.log(self.data.subsample_prob.values) # IPW
        linpred -= linpred.max()
        e_linpred = np.exp(linpred)

        # Initialize the meat matrix
        meat = np.zeros_like(self.z)
        
        # Initialize the A matrix        
        A = np.zeros([self.p, self.p])
        
        # Record the integral (reversely)
        int_dlambda = np.zeros(self.nuft + 1)
        int_z_bar_dlambda = np.zeros((self.nuft + 1, self.p))

        # Record the location of event time of each cases when inserted to the sequence of unique failure times (reversely)
        loc_event_time = np.zeros(self.N)
        
        xp0, xp1 = 0., 0.
        # Iterate backward through the unique failure times.
        for i in range(self.nuft)[::-1]:
            ix = self.risk_enter[i] # Indices of new cases entering the risk set.
            ixf = self.uft_ix[i] # Indices of cases that failed at this time
            loc_event_time[ix] = i + 1 # Note to plus 1 here

            xp0 += e_linpred[ix].sum()
            xp1 += (e_linpred[ix][:,None] * self.z[ix,:]).sum(0)
            z_bar = xp1/xp0
            zNt = self.z[ixf,:] - z_bar

            # Update the jump part of meat matrix
            meat[ixf,:] += zNt / self.data.subsample_prob.values[ixf][:,None]
            
            # Update the A matrix
            A += zNt.T.dot(zNt / self.data.subsample_prob.values[ixf][:,None])
            
            # Update the integral used to compute the compensator part of meat matrix
            dlambda = (1 / self.data.subsample_prob.values[ixf][:,None]).sum()/xp0
            int_dlambda[i+1] += dlambda
            int_z_bar_dlambda[i+1,:] += z_bar * dlambda

        # Compute the compensator part of meat matrix
        int_dlambda = int_dlambda.cumsum()
        int_z_bar_dlambda = int_z_bar_dlambda.cumsum(0)
        loc_event_time = loc_event_time.astype(int)
        meat -= self.z * (e_linpred * int_dlambda[loc_event_time])[:,None]
        meat += e_linpred[:,None] * int_z_bar_dlambda[loc_event_time, :]
        
        inv_A = inv(A)
        temp = inv_A.dot(meat.T).dot(meat).dot(inv_A)
        return np.sqrt(np.diag(temp))

    def Newton(self, beta0=None, stepsize=1e-2, max_iter=1e3, tol = 1e-1):
        if (beta0 == None).any():
            beta0 = np.zeros(self.p)
        beta = beta0.copy()

        for t in np.arange(max_iter):
            loss, grad, hess = self.grad_and_hess(beta)
            grad_norm = np.linalg.norm(grad, ord=2)
            print('iteration: %d, loss: %.2f, norm of gradient: %.2f' %(t+1, loss, grad_norm))

            # Stop iteration if the norm of gradient is smaller than tol.
            if (grad_norm <= tol):
                break

            # Newton update
            beta -= stepsize * inv(hess).dot(grad)

        # Estimated standard error
        se_plugin = self.se_est(beta)

        return [beta, se_plugin, int(t)]
    