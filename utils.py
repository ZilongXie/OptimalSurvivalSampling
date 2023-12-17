import numpy as np

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