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

# Search for the optimal threshold given a seqence of probabilies and subsampling rate 
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

# Generate censored data
class DataGeneration():
    def __init__(self, N, study_length=6):
        self.N = N
        self.study_length = study_length

    # Generate failure time from exponential distribution (time-invariant intensity)
    def generate_failure_time(self, hazard):
        failure_time = np.random.exponential(size=self.N, scale= 1/hazard)
        return failure_time

    # Generate censoring time from exponential distribution, controlled by parameter mu
    def generate_censoring_time(self, mu):
        temp = np.random.exponential(size=self.N, scale= 1/mu)
        result = np.clip(temp, None, self.study_length)  # set the length of study
        return result

    # Get observed follow-up time and number of controls
    def time_and_controls(self, failure_time, censoring_time):
        time_obs = np.minimum(failure_time, censoring_time)
        uncensored_indicator = (failure_time <= censoring_time).astype(int)
        censoring_rate = uncensored_indicator.mean()

        return time_obs, uncensored_indicator, censoring_rate

    def get_data(self, mu, hazard=None, other_failure_time=None):
        if other_failure_time is None:
            time_obs, uncensored_indicator, censoring_rate = self.time_and_controls(self.generate_failure_time(hazard), self.generate_censoring_time(mu))
        else:
            time_obs, uncensored_indicator, censoring_rate = self.time_and_controls(other_failure_time, self.generate_censoring_time(mu))
        delta = np.ones(self.N)
        pi = np.ones(self.N)
        result = pd.DataFrame({'time': time_obs, 'event': uncensored_indicator, 'subsample_indicator': delta, 'subsample_prob': pi}) # Use a dataframe to represent censored data
        return result

class FullSampleInference():
    def __init__(self, w, x, data, g='id', h='exp'):
        self.N = data.shape[0]
        self.p1 = w.shape[1]
        self.p2 = x.shape[1]
        self.p = self.p1 + self.p2
        self.g = g
        self.h = h
        z_DF = pd.concat([pd.DataFrame(w), pd.DataFrame(x)], axis=1)
        self.z_names = z_DF.columns # Save the names of covariates if exist
        data_concat = pd.concat([z_DF, data], axis=1) # Concat covariates and censored data
        self.w = data_concat.iloc[:,:self.p1].values # The W-covariate matrix
        self.x = data_concat.iloc[:,self.p1:self.p].values # The X-covariate matrix
        self.z = data_concat.iloc[:, :self.p].values  # The Z-covariate matrix
        self.data = data_concat.iloc[:,self.p:] # The censored dataset
        
        [self.uft, self.uft_ix, self.nuft,
         self.uet, self.uet_ix, self.nuet, self.uet_diff,
         self.f2e] = self.indices() # Quantities for computing survival objects
    
    def indices(self):
        # All failure times
        ift = np.flatnonzero(self.data.event.values == 1)
        ft = self.data.time.values[ift]

        # Unique failure times
        uft = np.unique(ft)
        nuft = len(uft)

        '''
        uft_ix: Indices of cases that fail at each unique failure time. Used for computing Z*dN(t).
        '''
        uft_map = dict([(x, i) for i,x in enumerate(uft)]) # Map unique failure time to its rank, len(uft_map) = nuft
        uft_ix = [[] for k in range(nuft)] # Map the rank of unique failure time to the indices of cases, len(uft_ix) = nuft
        for ix,ti in zip(ift,ft):
            uft_ix[uft_map[ti]].append(ix)
        
        # Unique event times.
        uet = np.unique(self.data.time.values) # sorted
        nuet = len(uet) 

        '''
        uet_diff: The difference of adjacent event times. len(uet_diff) = neft.
        '''
        uet_diff = np.diff(np.append([0.], uet)) # Set the left end as uet[0].
        
        '''
        uet_ix: Indices of cases that enter the risk set risk set at each unique event time. 
                Used for computing Z_bar(t)*Y(t).
        '''
        uet_map = dict([(x, i) for i,x in enumerate(uet)]) # Map unique event time to its rank, len(uet_map) = nuet
        uet_ix = [[] for k in range(nuet)] # Map the rank of unique event time to the indices of cases, len(uet_ix) = nuet
        for i,t in enumerate(self.data.time.values):
            uet_ix[uet_map[t]].append(i)
        
        '''
        f2e: Map the rank of each failtime in event time sequence to its rank (starts from 0) in failtime sequence. len(f2e) = nuft + 2.
        '''
        f2e = np.zeros(nuft+2).astype(int)
        for i,t in enumerate(uft):
            f2e[i+1] = uet_map[t]
        f2e[nuft+1] = nuet # Set the right end as nuet.
        
        return [uft, uft_ix, nuft, uet, uet_ix, nuet, uet_diff, f2e]

    def score_and_jacobian(self, theta):
        beta = theta[:self.p1]
        gamma = theta[self.p1:]

        wlinpred = np.dot(self.w, beta)
        if self.g == 'id':
            g_wlinpred = wlinpred
            dg_wlinpred = np.ones_like(wlinpred)
        if self.g == 'exp':
            g_wlinpred = np.exp(wlinpred)
            dg_wlinpred = g_wlinpred

        xlinpred = np.dot(self.x, gamma)
        if self.h == 'exp':
            xlinpred -= xlinpred.max()
            h_xlinpred = np.exp(xlinpred) # The utilization of h(X\gamma) involves a normalization step (computing z_bar).
            dh_xlinpred = np.exp(xlinpred) # The utilization of h^\prime(X\gamma) dinvolves a normalization step.

        hx0, hx1, hx2, hx3, gw0, gw1, gw2 = 0., 0., 0., 0., 0., 0., 0.,

        # Initialize the score
        score = -((self.data.time.values * g_wlinpred)[:, None] * self.z).sum(0)

        # Initialize the Jacobian
        jacobian = np.zeros((self.p, self.p))
        jacobian[:, :self.p1] -= self.z.T.dot((self.data.time.values * dg_wlinpred)[:,None] * self.w)

        # Initialize the 'meat' matrix in the sandwith formula
        meat = 0.

        # Iterate backward through the unique failure times.
        for i in range(self.nuft + 1)[::-1]:
            for j in range(self.f2e[i], self.f2e[i+1])[::-1]:
                ixe = self.uet_ix[j] # Indices of cases that enter the risk set at current event time.
                
                hx0 += h_xlinpred[ixe].sum()
                hx1 += (h_xlinpred[ixe][:,None] * self.z[ixe,:]).sum(0) 
                hx2 += (dh_xlinpred[ixe][:,None] * self.x[ixe,:]).sum(0) 
                hx3 += self.z[ixe,:].T.dot(dh_xlinpred[ixe][:,None] * self.x[ixe,:])

                gw0 += g_wlinpred[ixe].sum()
                gw1 += (g_wlinpred[ixe][:,None] * self.w[ixe,:]).sum(0)
                gw2 += (dg_wlinpred[ixe][:,None] * self.w[ixe,:]).sum(0)

                # Update score
                z_bar = hx1/hx0
                score += self.uet_diff[j] * z_bar * gw0

                # Update Jacobian
                dz_bar = hx3/hx0 - z_bar[:, None].dot((hx2/hx0)[None, :])
                jacobian[:, :self.p1] += self.uet_diff[j] * z_bar[:,None].dot(gw2[None,:])
                jacobian[:, self.p1:] += self.uet_diff[j] * gw0 * dz_bar

            if i > 0:
                # \int...dN(t)
                ixf = self.uft_ix[i-1]
                zNt = self.z[ixf,:] - z_bar # Note here we use Z_i which failed at current failure time. (i:dN_i(t) > 0)

                # Update score
                score += zNt.sum(0)

                # Update Jacobian
                jacobian[:, self.p1:] -= dz_bar * len(ixf)

                # Update meat
                meat += zNt.T.dot(zNt)

        return [score, jacobian, meat]

    def Newton(self, theta0=None, stepsize=1e-2, max_iter=1e3, tol = 1e-1):
        if (theta0 == None).any():
            theta0 = np.zeros(self.p)
        theta = theta0.copy()

        for t in np.arange(max_iter):
            score, jacobian, meat = self.score_and_jacobian(theta)
            score_norm = np.linalg.norm(score, ord=2)
            print('iteration: ' + str(int(t + 1)) + ', ' + 'norm of score: '+str(score_norm))

            # Stop iteration if the norm of score is smaller than tol.
            if (score_norm <= tol):
                break

            # Newton update
            theta -= stepsize * inv(jacobian).dot(score)

            if (np.isnan(score_norm)):
                break

        # The plug-in estimate of standard error
        temp = inv(jacobian).dot(meat).dot(inv(jacobian).T)
        se_plugin = np.sqrt(np.diag(temp))

        return [theta, jacobian, se_plugin, int(t)]



class GamSampling():
    def __init__(self, w, x, data, beta, gamma, g='id', h='exp'):
        self.N = data.shape[0]
        self.p1 = w.shape[1]
        self.p2 = x.shape[1]
        self.p = self.p1 + self.p2
        self.beta = beta
        self.gamma = gamma
        self.g = g
        self.h = h
        z_DF = pd.concat([pd.DataFrame(w), pd.DataFrame(x)], axis=1)
        self.z_names = z_DF.columns # Save the names of covariates if exist
        self.w = z_DF.iloc[:,:self.p1].values # The W-covariate matrix
        self.x = z_DF.iloc[:,self.p1:self.p].values # The X-covariate matrix
        self.z = z_DF.values  # The Z-covariate matrix
        self.data = data# The censored dataset
        
        [self.uft, self.uft_ix, self.nuft,
         self.uet, self.uet_ix, self.nuet, self.uet_diff,
         self.f2e] = self.indices() # Quantities for computing survival objects
    
    def indices(self):
        # All failure times
        ift = np.flatnonzero(self.data.event.values == 1)
        ft = self.data.time.values[ift]

        # Unique failure times
        uft = np.unique(ft)
        nuft = len(uft)

        '''
        uft_ix: Indices of cases that fail at each unique failure time. Used for computing Z*dN(t).
        '''
        uft_map = dict([(x, i) for i,x in enumerate(uft)]) # Map unique failure time to its rank, len(uft_map) = nuft
        uft_ix = [[] for k in range(nuft)] # Map the rank of unique failure time to the indices of cases, len(uft_ix) = nuft
        for ix,ti in zip(ift,ft):
            uft_ix[uft_map[ti]].append(ix)
        
        # Unique event times.
        uet = np.unique(self.data.time.values) # sorted
        nuet = len(uet) 

        '''
        uet_diff: The difference of adjacent event times. len(uet_diff) = neft.
        '''
        uet_diff = np.diff(np.append([0.], uet)) # Set the left end as uet[0].
        
        '''
        uet_ix: Indices of cases that enter the risk set risk set at each unique event time. 
                Used for computing Z_bar(t)*Y(t).
        '''
        uet_map = dict([(x, i) for i,x in enumerate(uet)]) # Map unique event time to its rank, len(uet_map) = nuet
        uet_ix = [[] for k in range(nuet)] # Map the rank of unique event time to the indices of cases, len(uet_ix) = nuet
        for i,t in enumerate(self.data.time.values):
            uet_ix[uet_map[t]].append(i)
        
        '''
        f2e: Map the rank of each failtime in event time sequence to its rank (starts from 0) in failtime sequence. len(f2e) = nuft + 2.
        '''
        f2e = np.zeros(nuft+2).astype(int)
        for i,t in enumerate(uft):
            f2e[i+1] = uet_map[t]
        f2e[nuft+1] = nuet # Set the right end as nuet.
        
        return [uft, uft_ix, nuft, uet, uet_ix, nuet, uet_diff, f2e]

    def pi(self):
        wlinpred = np.dot(self.w, self.beta)
        if self.g == 'id':
            g_wlinpred = wlinpred
            dg_wlinpred = np.ones_like(wlinpred)
        if self.g == 'exp':
            g_wlinpred = np.exp(wlinpred)
            dg_wlinpred = g_wlinpred

        xlinpred = np.dot(self.x, self.gamma)
        if self.h == 'exp':
            xlinpred -= xlinpred.max()
            h_xlinpred = np.exp(xlinpred) # The utilization of h(X\gamma) involves a normalization step (computing z_bar).
            dh_xlinpred = np.exp(xlinpred) # The utilization of h^\prime(X\gamma) dinvolves a normalization step.

        hx0, hx1, hx2, hx3, gw0, gw1, gw2 = 0., 0., 0., 0., 0., 0., 0.

        # Initialize the Jacobian
        jacobian = np.zeros((self.p, self.p))
        jacobian[:, :self.p1] -= self.z.T.dot((self.data.time.values * dg_wlinpred)[:,None] * self.w)

        # Initialize the 'meat' matrix in the sandwith formula
        meat = np.zeros((self.N, self.p))
        meat_vec = np.zeros(self.N)
        meat_vec -= self.data.time.values * g_wlinpred
        
        # Record the integral (reversely)
        int_dlambda = np.zeros(self.nuet)
        int_z_bar_dt = np.zeros((self.nuet, self.p))
        int_z_bar_dlambda = np.zeros((self.nuet, self.p))

        # Record the rank of event time of each cases (reversely)
        rank_event_time = np.zeros(self.N)

        # Iterate backward through the unique failure times.
        for i in range(self.nuft + 1)[::-1]:
            for j in range(self.f2e[i], self.f2e[i+1])[::-1]:
                ixe = self.uet_ix[j] # Indices of cases that enter the risk set at current event time.
                
                hx0 += h_xlinpred[ixe].sum()
                hx1 += (h_xlinpred[ixe][:,None] * self.z[ixe,:]).sum(0)
                hx2 += (dh_xlinpred[ixe][:,None] * self.x[ixe,:]).sum(0) 
                hx3 += self.z[ixe,:].T.dot(dh_xlinpred[ixe][:,None] * self.x[ixe,:])

                gw0 += g_wlinpred[ixe].sum()
                gw1 += (g_wlinpred[ixe][:,None] * self.w[ixe,:]).sum(0)
                gw2 += (dg_wlinpred[ixe][:,None] * self.w[ixe,:]).sum(0)

                # Update score
                z_bar = hx1/hx0

                # Update Jacobian
                dz_bar = hx3/hx0 - z_bar[:, None].dot((hx2/hx0)[None, :])
                jacobian[:, :self.p1] += self.uet_diff[j] * z_bar[:,None].dot(gw2[None,:])
                jacobian[:, self.p1:] += self.uet_diff[j] * gw0 * dz_bar
                
                # Update integral
                dlambda = -gw0/hx0
                int_dlambda[j] += self.uet_diff[j] * dlambda
                int_z_bar_dt[j,:] += (self.uet_diff[j] * z_bar)
                int_z_bar_dlambda[j,:] += (self.uet_diff[j] * z_bar * dlambda)
                rank_event_time[ixe] = j

            if i > 0:
                # \int...dN(t)
                ixf = self.uft_ix[i-1]
                zNt = self.z[ixf,:] - z_bar # Note here we use Z_i which failed at current failure time. (i:dN_i(t) > 0)

                # Update Jacobian
                jacobian[:, self.p1:] -= dz_bar * len(ixf)

                # Update meat
                meat[ixf,:] += zNt

                # Update int_z_bar_dlambda
                int_dlambda[j] += len(ixf)/hx0
                int_z_bar_dlambda[j,:] += z_bar * len(ixf)/hx0

        int_dlambda = int_dlambda.cumsum()
        int_z_bar_dt = int_z_bar_dt.cumsum(0)
        int_z_bar_dlambda = int_z_bar_dlambda.cumsum(0)
        
        rank_event_time = rank_event_time.astype(int)
        meat_vec -= h_xlinpred * int_dlambda[rank_event_time]
        meat += self.z * meat_vec[:, None]
        meat += g_wlinpred[:,None] * int_z_bar_dt[rank_event_time, :]
        meat += h_xlinpred[:,None] * int_z_bar_dlambda[rank_event_time, :]

        jacobian /= self.N
        invA = inv(jacobian)
        temp = meat.dot(invA)
        trace_of_var =  np.einsum('ij,ij->i', temp, temp) # N*1 vector
        result = np.sqrt(trace_of_var)

        return result
    
    def get_subsample_data(self, subsampling_rate):
        temp1 = self.pi()

        # Optimal subsampling
        c1 = find_c(temp1, subsampling_rate)
        pi = np.clip(c1 * np.array(temp1), None, 1)

        sub_data = self.data.copy()
        result = np.random.binomial(n=1, p=pi, size=self.N)
        sub_data.subsample_prob = pi
        sub_data.subsample_indicator = result
        
        sub_data = sub_data.loc[sub_data.subsample_indicator == 1]
        sub_z = pd.DataFrame(self.z.copy()[result == 1],
                            index = sub_data.index,
                            columns = self.z_names
                            )
        sub_w = sub_z.iloc[:, :self.p1]
        sub_x = sub_z.iloc[:, self.p1:]

        return sub_w, sub_x, sub_data


class SubSampleInference():
    def __init__(self, w, x, data, g='id', h='exp'):
        self.N = data.shape[0]
        self.p1 = w.shape[1]
        self.p2 = x.shape[1]
        self.p = self.p1 + self.p2
        self.g = g
        self.h = h
        z_DF = pd.concat([pd.DataFrame(w), pd.DataFrame(x)], axis=1)
        self.z_names = z_DF.columns # Save the names of covariates if exist
        data_concat = pd.concat([z_DF, data], axis=1) # Concat covariates and censored data
        self.w = data_concat.iloc[:,:self.p1].values # The W-covariate matrix
        self.x = data_concat.iloc[:,self.p1:self.p].values # The X-covariate matrix
        self.z = data_concat.iloc[:, :self.p].values  # The Z-covariate matrix
        self.data = data_concat.iloc[:,self.p:] # The censored dataset
        
        [self.uft, self.uft_ix, self.nuft,
         self.uet, self.uet_ix, self.nuet, self.uet_diff,
         self.f2e] = self.indices() # Quantities for computing survival objects
    
    def indices(self):
        # All failure times
        ift = np.flatnonzero(self.data.event.values == 1)
        ft = self.data.time.values[ift]

        # Unique failure times
        uft = np.unique(ft)
        nuft = len(uft)

        '''
        uft_ix: Indices of cases that fail at each unique failure time. Used for computing Z*dN(t).
        '''
        uft_map = dict([(x, i) for i,x in enumerate(uft)]) # Map unique failure time to its rank, len(uft_map) = nuft
        uft_ix = [[] for k in range(nuft)] # Map the rank of unique failure time to the indices of cases, len(uft_ix) = nuft
        for ix,ti in zip(ift,ft):
            uft_ix[uft_map[ti]].append(ix)
        
        # Unique event times.
        uet = np.unique(self.data.time.values) # sorted
        nuet = len(uet) 

        '''
        uet_diff: The difference of adjacent event times. len(uet_diff) = neft.
        '''
        uet_diff = np.diff(np.append([0.], uet)) # Set the left end as uet[0].
        
        '''
        uet_ix: Indices of cases that enter the risk set risk set at each unique event time. 
                Used for computing Z_bar(t)*Y(t).
        '''
        uet_map = dict([(x, i) for i,x in enumerate(uet)]) # Map unique event time to its rank, len(uet_map) = nuet
        uet_ix = [[] for k in range(nuet)] # Map the rank of unique event time to the indices of cases, len(uet_ix) = nuet
        for i,t in enumerate(self.data.time.values):
            uet_ix[uet_map[t]].append(i)
        
        '''
        f2e: Map the rank of each failtime in event time sequence to its rank (starts from 0) in failtime sequence. len(f2e) = nuft + 2.
        '''
        f2e = np.zeros(nuft+2).astype(int)
        for i,t in enumerate(uft):
            f2e[i+1] = uet_map[t]
        f2e[nuft+1] = nuet # Set the right end as nuet.
        
        return [uft, uft_ix, nuft, uet, uet_ix, nuet, uet_diff, f2e]

    def score_and_jacobian(self, theta):
        beta = theta[:self.p1]
        gamma = theta[self.p1:]

        # Note here we scaled h and g by IPW
        wlinpred = np.dot(self.w, beta)
        if self.g == 'id':
            g_wlinpred = wlinpred / self.data.subsample_prob.values
            dg_wlinpred = np.ones_like(wlinpred) / self.data.subsample_prob.values
        if self.g == 'exp':
            g_wlinpred = np.exp(wlinpred) / self.data.subsample_prob.values
            dg_wlinpred = g_wlinpred / self.data.subsample_prob.values

        xlinpred = np.dot(self.x, gamma)
        if self.h == 'exp':
            xlinpred -= np.log(self.data.subsample_prob.values) # IPW
            xlinpred -= xlinpred.max()
            h_xlinpred = np.exp(xlinpred) # The utilization of h(X\gamma) involves a normalization step (computing z_bar).
            dh_xlinpred = np.exp(xlinpred) # The utilization of h^\prime(X\gamma) involves a normalization step.
        
        hx0, hx1, hx2, hx3, gw0, gw1, gw2 = 0., 0., 0., 0., 0., 0., 0.

        # Initialize the score
        score = -((self.data.time.values * g_wlinpred)[:, None] * self.z).sum(0)

        # Initialize the Jacobian
        jacobian = np.zeros((self.p, self.p))
        jacobian[:, :self.p1] -= self.z.T.dot((self.data.time.values * dg_wlinpred)[:,None] * self.w)

        # Iterate backward through the unique failure times.
        for i in range(self.nuft + 1)[::-1]:
            for j in range(self.f2e[i], self.f2e[i+1])[::-1]:
                ixe = self.uet_ix[j] # Indices of cases that enter the risk set at current event time.
                
                hx0 += h_xlinpred[ixe].sum()
                hx1 += (h_xlinpred[ixe][:,None] * self.z[ixe,:]).sum(0) 
                hx2 += (dh_xlinpred[ixe][:,None] * self.x[ixe,:]).sum(0) 
                hx3 += self.z[ixe,:].T.dot(dh_xlinpred[ixe][:,None] * self.x[ixe,:])

                gw0 += g_wlinpred[ixe].sum()
                gw1 += (g_wlinpred[ixe][:,None] * self.w[ixe,:]).sum(0)
                gw2 += (dg_wlinpred[ixe][:,None] * self.w[ixe,:]).sum(0)

                # Update score
                z_bar = hx1/hx0
                score += self.uet_diff[j] * z_bar * gw0

                # Update Jacobian
                dz_bar = hx3/hx0 - z_bar[:, None].dot((hx2/hx0)[None, :])
                jacobian[:, :self.p1] += self.uet_diff[j] * z_bar[:,None].dot(gw2[None,:])
                jacobian[:, self.p1:] += self.uet_diff[j] * gw0 * dz_bar

            if i > 0:
                # \int...dN(t)
                ixf = self.uft_ix[i-1]
                zNt = (self.z[ixf,:] - z_bar) / self.data.subsample_prob.values[ixf][:,None] # Note here we scaled zNt by IPW
                
                # Update score
                score += zNt.sum(0)

                # Update Jacobian
                jacobian[:, self.p1:] -= dz_bar * (1 / self.data.subsample_prob.values[ixf]).sum()

        return [score, jacobian]

    def meat(self, theta):
        beta = theta[:self.p1]
        gamma = theta[self.p1:]

        # Note here we scaled h and g by IPW
        wlinpred = np.dot(self.w, beta)
        if self.g == 'id':
            g_wlinpred = wlinpred / self.data.subsample_prob.values
            dg_wlinpred = np.ones_like(wlinpred) / self.data.subsample_prob.values
        if self.g == 'exp':
            g_wlinpred = np.exp(wlinpred) / self.data.subsample_prob.values
            dg_wlinpred = g_wlinpred / self.data.subsample_prob.values

        xlinpred = np.dot(self.x, gamma)
        if self.h == 'exp':
            xlinpred -= np.log(self.data.subsample_prob.values) # IPW
            xlinpred -= xlinpred.max()
            h_xlinpred = np.exp(xlinpred) # The utilization of h(X\gamma) involves a normalization step (computing z_bar).
            dh_xlinpred = np.exp(xlinpred) # The utilization of h^\prime(X\gamma) involves a normalization step.
        
        hx0, hx1, hx2, hx3, gw0, gw1, gw2 = 0., 0., 0., 0., 0., 0., 0.
        
        # Initialize the 'meat' matrix in the sandwith formula
        meat = np.zeros((self.N, self.p))
        meat_vec = np.zeros(self.N)
        meat_vec -= self.data.time.values * g_wlinpred
        
        # Record the integral (reversely)
        int_dlambda = np.zeros(self.nuet)
        int_z_bar_dt = np.zeros((self.nuet, self.p))
        int_z_bar_dlambda = np.zeros((self.nuet, self.p))

        # Record the rank of event time of each cases (reversely)
        rank_event_time = np.zeros(self.N)

        # Iterate backward through the unique failure times.
        for i in range(self.nuft + 1)[::-1]:
            for j in range(self.f2e[i], self.f2e[i+1])[::-1]:
                ixe = self.uet_ix[j] # Indices of cases that enter the risk set at current event time.

                hx0 += h_xlinpred[ixe].sum()
                hx1 += (h_xlinpred[ixe][:,None] * self.z[ixe,:]).sum(0)
                hx2 += (dh_xlinpred[ixe][:,None] * self.x[ixe,:]).sum(0) 
                hx3 += self.z[ixe,:].T.dot(dh_xlinpred[ixe][:,None] * self.x[ixe,:])

                gw0 += g_wlinpred[ixe].sum()
                gw1 += (g_wlinpred[ixe][:,None] * self.w[ixe,:]).sum(0)
                gw2 += (dg_wlinpred[ixe][:,None] * self.w[ixe,:]).sum(0)

                # Update score
                z_bar = hx1/hx0
 
                # Update integral
                dlambda = -gw0/hx0
                int_dlambda[j] += self.uet_diff[j] * dlambda
                int_z_bar_dt[j,:] += (self.uet_diff[j] * z_bar)
                int_z_bar_dlambda[j,:] += (self.uet_diff[j] * z_bar * dlambda)
                rank_event_time[ixe] = j

            if i > 0:
                # \int...dN(t)
                ixf = self.uft_ix[i-1]
                zNt = self.z[ixf,:] - z_bar # Note here we use Z_i which failed at current failure time. (i:dN_i(t) > 0)

                # Update meat
                meat[ixf,:] += zNt / self.data.subsample_prob.values[ixf][:,None]

                # Update int_z_bar_dlambda
                int_dlambda[j] += (1 / self.data.subsample_prob.values[ixf]).sum()/hx0
                int_z_bar_dlambda[j,:] += z_bar * (1 / self.data.subsample_prob.values[ixf]).sum()/hx0

        int_dlambda = int_dlambda.cumsum()
        int_z_bar_dt = int_z_bar_dt.cumsum(0)
        int_z_bar_dlambda = int_z_bar_dlambda.cumsum(0)
        
        rank_event_time = rank_event_time.astype(int)
        meat_vec -= h_xlinpred * int_dlambda[rank_event_time]
        meat += self.z * meat_vec[:, None]
        meat += g_wlinpred[:,None] * int_z_bar_dt[rank_event_time, :]
        meat += h_xlinpred[:,None] * int_z_bar_dlambda[rank_event_time, :]

        return meat.T.dot(meat)

    def Newton(self, theta0=None, stepsize=1e-2, max_iter=1e3, tol = 1e-1):
        if (theta0 == None).any():
            theta0 = np.zeros(self.p)
        theta = theta0.copy()

        for t in np.arange(max_iter):
            score, jacobian = self.score_and_jacobian(theta)
            score_norm = np.linalg.norm(score, ord=2)
            print('iteration: ' + str(int(t + 1)) + ', ' + 'norm of score: '+str(np.linalg.norm(score, ord=2)))

            # Stop iteration if the norm of score is smaller than tol.
            if (score_norm <= tol):
                break

            # Newton update
            theta -= stepsize * inv(jacobian).dot(score)

        # Compute meat matrix
        meat = self.meat(theta)

        # The plug-in estimate of standard error
        temp = inv(jacobian).dot(meat).dot(inv(jacobian).T)
        se_plugin = np.sqrt(np.diag(temp))

        return [theta, jacobian, se_plugin, int(t)]