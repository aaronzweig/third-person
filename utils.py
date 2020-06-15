import numpy as np
from mdp import DiscountedMDP, MRP, random_MDP, random_MDP_forward_reward
from numpy import sqrt

def full_occupancy(mdp, phi):
    d = (mdp | phi).d()
    return np.expand_dims(d,1) * phi

def empirical_occupancy(mdp, phi, length, verbose = False):
    gamma = mdp.gamma
    S, A = phi.shape
    d = np.zeros((S,A))
    step = 0
    
    for s,a,r,sp in mdp.run(phi):
        if verbose:
            print(s, a, r, sp)
        if step >= length:
            break
        d[s][a] += 1
        step += 1
        
    return d / np.sum(d)

def draw_samples(mdp, phi, length):
    gamma = mdp.gamma
    S, A = phi.shape
    step = 0
    samples = []
    
    for s,a,r,sp in mdp.run(phi):
        if step >= length:
            break
        step += 1
        samples.append((s,sp))
        
    return samples

def tv(d_1, d_2):
    return 0.5 * np.sum(np.abs(d_1-d_2))

def state_occupancy(mdp, phi):
    return (mdp | phi).d()




def to_policy_matrix(phi):
    S, A = phi.shape
    diags = []
    for a in range(A):
        Phi_a = np.diag(phi[:,a])
        diags.append(Phi_a)
    return np.vstack(diags)

def from_policy_matrix(Phi):
    SA, S = Phi.shape
    A = int(SA / S)

    phi = np.zeros((S, A))
    for a in range(A):
        Phi_a = Phi[S*a:S*(a+1),:]
        phi[:,a] = np.diag(Phi_a)
    return phi




def subindex(M, I):
    sub = M[I,:]
    return sub[:,I]

def invsqrt(D):
    d = np.diag(D)
    d = 1.0 / sqrt(d)
    return np.diag(d)