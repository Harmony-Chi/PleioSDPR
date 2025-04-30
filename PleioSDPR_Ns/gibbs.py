import numpy as np
from scipy import stats, special, linalg
from collections import Counter
from joblib import Parallel, delayed
from sklearn import linear_model
import math

def initial_state(data1, data2, M, N1, N2, idx1_shared, idx2_shared, ld_boundaries1, ld_boundaries2,alpha=1.0, a0k=.5, b0k=.5):
    num_clusters = [M[1]+1, M[2]+1, sum(M)]
    cluster_ids = [range(num_clusters[0]), range(num_clusters[1]), range(num_clusters[2])]

    state = {
        'num_clusters_': num_clusters,
        'cluster_ids_': cluster_ids,
        'M':M,
        
        'beta_margin1_': data1,
        'beta_margin2_': data2,
        'N1_': N1,
        'N2_': N2,
        'beta1': np.zeros(len(data1)),
        'beta2': np.zeros(len(data2)),
        
        'b1': np.zeros(len(data1)),
        'b2': np.zeros(len(data2)),
        'b1_': np.zeros(len(data1)),
        'b2_': np.zeros(len(data2)),
        
        'cluster_var1': [ np.array([0.0]*M[3])],
        'cluster_var2':[ np.array([0.0]*M[3])],
        'cluster_rho':[ np.array([0.0]*M[3])],
        'delta': [np.array([0.0001]*M[3])],
        'lambda': [np.array([0.0001]*M[3])],
        
        'cluster_var_specific': [np.array([0]*(num_clusters[0])),
                                 np.array([0]*(num_clusters[1])), 
                                 np.array([0]*(num_clusters[2]-M[3]))],
        'gamma': [np.array([0.5]* num_clusters[0]), np.array([0.5]*num_clusters[0])],
        'theta': [np.array([0.5]* num_clusters[1]), np.array([0.5]*num_clusters[1])],
        
        'hyperparameters_': {
            "a0k": a0k,
            "b0k": b0k,
            "a0": 0.1,
            "b0": 0.1,
            "iw_b1":0.5,
            "iw_b2":0.5,
            "iw_b3":0.5,
            "iw_b4":0.5,
            "v":0.5001,
            "iw_phi1":1,
            "iw_phi2":1,
            "iw_phi3":1,
            "iw_phi4":1
        },
        'suffstats': [np.array([0]*num_clusters[0]), np.array([0]*num_clusters[1]), np.array([0]*num_clusters[2])],
        'assignment1': np.random.randint(num_clusters[2], size=len(data1)),
        'assignment2': np.array([0]*len(data2)),
        'population': np.array([0]*4),
        
        'alpha': [np.array([alpha]), np.array([alpha]), np.array([alpha]*4)],
        'pi': [np.array([alpha / num_clusters[0]]*num_clusters[0]), np.array([alpha / num_clusters[1]]*num_clusters[1]), np.array([alpha / num_clusters[2]]*num_clusters[2])],
        'pi_pop': np.array([.25, .25, .25, .25]),
        'V':  [np.array([0]*num_clusters[0]), np.array([0]*num_clusters[1]), [np.array([0]*M[i]) for i in range(len(M))]],
        'varg1': np.array([0.0]*len(ld_boundaries1)),
        'varg2': np.array([0.0]*len(ld_boundaries1)),
        'covg12': np.array([0.0]*len(ld_boundaries1)),
        'h2_1': 0,
        'h2_2': 0
    }

    # define indexes
    state['population'][0] = M[0] #1
    state['population'][1] = M[0]+M[1] #1001
    state['population'][2] = M[0]+M[1]+M[2] #2001
    state['population'][3] = M[0]+M[1]+M[2]+M[3] #3001

    tmp1 = []; tmp2 = []; tmp3 = []; tmp4 = []
    for j in range(len(ld_boundaries1)):
        start_i1 = ld_boundaries1[j][0]
        end_i1 = ld_boundaries1[j][1]
        start_i2 = ld_boundaries2[j][0]
        end_i2 = ld_boundaries2[j][1]
        tmp1.append(np.setdiff1d(np.array(range(end_i1-start_i1)), idx1_shared[j])+start_i1)
        tmp2.append(np.setdiff1d(np.array(range(end_i2-start_i2)), idx2_shared[j])+start_i2)
        tmp3.append(idx1_shared[j]+start_i1)
        tmp4.append(idx2_shared[j]+start_i2)
        state['assignment2'][idx2_shared[j]+start_i2] = state['assignment1'][idx1_shared[j]+start_i1]
    
    state['idx_pop1'] = np.concatenate(tmp1)
    state['idx_pop2'] = np.concatenate(tmp2)
    state['idx_shared1'] = np.concatenate(tmp3)
    state['idx_shared2'] = np.concatenate(tmp4)

    state['assignment1'][state['idx_pop1']] = np.random.randint(low=1, high=M[1]+1, size=len(state['idx_pop1']))
    state['assignment2'][state['idx_pop2']] = np.random.randint(low=1, high=M[1]+1, size=len(state['idx_pop2']))

    return state

def update_suffstats(state):
    # shared variants between two GWAS
    assn = state['assignment1'][state['idx_shared1']]
    suff_stats_shared = dict(Counter(assn))
    suff_stats_shared.update(dict.fromkeys(np.setdiff1d(range(state['num_clusters_'][2]), suff_stats_shared.keys()), 0))
    
    # trait1 specific variants
    assn = state['assignment1'][state['idx_pop1']]
    suff_stats_pop1 = dict(Counter(assn))
    suff_stats_pop1.update(dict.fromkeys(np.setdiff1d(range(state['num_clusters_'][0]), suff_stats_pop1.keys()), 0))

    # trait2 specific variants
    assn = state['assignment2'][state['idx_pop2']]
    suff_stats_pop2 = dict(Counter(assn))
    suff_stats_pop2.update(dict.fromkeys(np.setdiff1d(range(state['num_clusters_'][1]), suff_stats_pop2.keys()), 0))

    return [suff_stats_pop1, suff_stats_pop2, suff_stats_shared]


### Sample the the variance of the second and third components for triat1-specific SNPs, trait2-specific SNPs and shared SNPs.
def sample_sigma2(state):
    M = state['M']
    b = np.zeros(state['num_clusters_'][2]-M[3])
    a = np.array(list(state['suffstats'][2].values())[:(state['num_clusters_'][2]-M[3])] ) / 2.0 + state['hyperparameters_']['a0k'] 

    table1 = [[] for i in range(state['num_clusters_'][2])]
    for i in range(len(state['assignment1'][state['idx_shared1']])):
        table1[state['assignment1'][state['idx_shared1']][i]].append(i)

    table2 = [[] for i in range(state['num_clusters_'][2])]
    for i in range(len(state['assignment2'][state['idx_shared2']])):
        table2[state['assignment2'][state['idx_shared2']][i]].append(i)

    for i in range(M[0], M[1]+M[0]):
        beta = state['beta1'][state['idx_shared1']][table1[i]]
        b[i] = np.sum(beta**2) / 2.0 + state['gamma'][1][i]

    for i in range(M[1]+M[0],M[2]+M[1]+M[0]):
        beta = state['beta2'][state['idx_shared2']][table2[i]]
        b[i] = np.sum(beta**2) / 2.0 + state['theta'][1][i-(M[1])]

    out_shared = np.array([0.0]*(state['num_clusters_'][2]-M[3]))
    out_shared[1:] = stats.invgamma(a=a[1:], scale=b[1:]).rvs()

    # trait1 specific variants
    b = np.zeros(state['num_clusters_'][0])
    a = np.array(list(state['suffstats'][0].values()) ) / 2.0 + state['hyperparameters_']['a0k']
    # The table records that the i th trait1-specific SNP in each component, e.g. table1[0] records the sequences of 
    # trait1-specific SNPs assigned to component 0
    table1 = [[] for i in range(state['num_clusters_'][0])]
    for i in range(len(state['assignment1'][state['idx_pop1']])):
        table1[state['assignment1'][state['idx_pop1']][i]].append(i)

    for i in range(state['num_clusters_'][0]):
        beta=state['beta1'][state['idx_pop1']][table1[i]]
        b[i] = np.sum(beta**2) / 2.0 + state['gamma'][0][i]
        
    out_pop1 = np.array([0.0]*state['num_clusters_'][0])
    out_pop1[1:] = stats.invgamma(a=a[1:], scale=b[1:]).rvs()

    # trait2 specific variants
    b = np.zeros(state['num_clusters_'][1])
    a = np.array(list(state['suffstats'][1].values()) ) / 2.0 + state['hyperparameters_']['b0k']
    table2 = [[] for i in range(state['num_clusters_'][1])]
    for i in range(len(state['assignment2'][state['idx_pop2']])):
        table2[state['assignment2'][state['idx_pop2']][i]].append(i)

    for i in range(state['num_clusters_'][1]):
        beta =  state['beta2'][state['idx_pop2']][table2[i]]
        b[i] = np.sum(beta**2) / 2.0 + state['theta'][0][i]
        
    out_pop2 = np.array([0.0]*state['num_clusters_'][1])
    out_pop2[1:] = stats.invgamma(a=a[1:], scale=b[1:]).rvs()

    return [out_pop1, out_pop2, out_shared]


def sample_gamma(state):
    sigma1k_2= state["cluster_var_specific"][0][1:state["population"][1]]
    a =  state['hyperparameters_']['iw_b3']
    b = state['hyperparameters_']['iw_phi3'] + 1/sigma1k_2
    state['gamma'][0][1:] = stats.gamma(a=a, scale=1.0/b).rvs()
    
    sigma1k_2= state["cluster_var_specific"][2][1:state["population"][1]]
    a =  state['hyperparameters_']['iw_b3']
    b = state['hyperparameters_']['iw_phi3'] + 1/sigma1k_2
    state['gamma'][1][1:] = stats.gamma(a=a, scale=1.0/b).rvs()

def sample_theta(state):
    sigma2k_2= state["cluster_var_specific"][1][1:state["population"][1]]
    a =  state['hyperparameters_']['iw_b4']
    b = state['hyperparameters_']['iw_phi4'] + 1/sigma2k_2
    state['theta'][0][1:] = stats.gamma(a=a, scale=1.0/b).rvs()
    
    sigma2k_2= state["cluster_var_specific"][2][state["population"][1]:state["population"][2]]
    a =  state['hyperparameters_']['iw_b4']
    b = state['hyperparameters_']['iw_phi4'] + 1/sigma2k_2
    state['theta'][1][1:] = stats.gamma(a=a, scale=1.0/b).rvs()
    

def sample_IW(state):
    M=state['M']
    b11 = np.zeros(M[3])
    b12 = np.zeros(M[3])
    b22 = np.zeros(M[3])
    a = np.array(list(state['suffstats'][2].values())[(state['num_clusters_'][2]-M[3]):])  + 2* state['hyperparameters_']['v'] +1
    
    table1 = [[] for i in range(state['num_clusters_'][2])]
    for i in range(len(state['assignment1'][state['idx_shared1']])):
        table1[state['assignment1'][state['idx_shared1']][i]].append(i)

    table2 = [[] for i in range(state['num_clusters_'][2])]
    for i in range(len(state['assignment2'][state['idx_shared2']])):
        table2[state['assignment2'][state['idx_shared2']][i]].append(i)

    for i in range(state['population'][2],state['population'][3]):
        beta1 = state['beta1'][state['idx_shared1']][table1[i]]
        beta2 = state['beta2'][state['idx_shared2']][table2[i]]
        b11[i-(state['population'][2])] = np.sum( (beta1**2 )) + 4*state['hyperparameters_']['v']*state['delta'][0][i-(state['population'][2])]
        b12[i-(state['population'][2])] = np.sum( (beta1*beta2))
        b22[i-(state['population'][2])] = np.sum( (beta2**2 )) + 4*state['hyperparameters_']['v']*state['lambda'][0][i-(state['population'][2])]

    out_var1 = np.array([0.0]*(M[3]))
    out_var2 = np.array([0.0]*(M[3]))
    out_gcov = np.array([0.0]*(M[3]))
    params = [(a[i], [[b11[i], b12[i]], [b12[i], b22[i]]]) 
          for i in range(0, M[3])]
    samples = [stats.invwishart(df=param[0], scale=param[1]).rvs() for param in params]
    out_var1=np.array([sample[0,0] for sample in samples])
    out_var2=np.array([sample[1,1] for sample in samples])
    out_gcov=np.array([sample[0,1] for sample in samples])
    return [out_var1, out_var2, out_gcov]



def sample_delta(state):
    sigma3k_2= state["cluster_var1"][0]
    sigma4k_2= state["cluster_var2"][0]
    r= state["cluster_rho"][0]/np.sqrt(sigma3k_2*sigma4k_2)
    a = state['hyperparameters_']['v'] + 0.5 + state['hyperparameters_']['iw_b1']
    b = state['hyperparameters_']['iw_phi1'] + 2*state['hyperparameters_']['v']/(sigma3k_2-r**2*sigma3k_2)
    state['delta'][0] = stats.gamma(a=a, scale=1.0/b).rvs()

def sample_lambda(state):
    sigma3k_2= state["cluster_var1"][0]
    sigma4k_2= state["cluster_var2"][0]
    r= state["cluster_rho"][0]/np.sqrt(sigma3k_2*sigma4k_2)
    a = state['hyperparameters_']['v'] + 0.5 + state['hyperparameters_']['iw_b2']
    b = state['hyperparameters_']['iw_phi2'] + 2*state['hyperparameters_']['v']/(sigma4k_2-r**2*sigma4k_2)
    state['lambda'][0] = stats.gamma(a=a, scale=1.0/b).rvs()
    
    
def calc_b(j, state, ld_boundaries1, ld_boundaries2, idx1_shared, idx2_shared):
    start_i1 = ld_boundaries1[j][0]
    end_i1 = ld_boundaries1[j][1]
    start_i2 = ld_boundaries2[j][0]
    end_i2 = ld_boundaries2[j][1]
    
    
    N1 = state['N1_']; N2 = state['N2_'];s=state['ovp_pcor']
    s1=N1/(1-s**2);s2=-s*np.sqrt(N1*N2)/(1-s**2);s3=N2/(1-s**2)
    
    b1 = np.dot(state['R1'][j], state['beta_margin1_'][start_i1:end_i1]) -  \
    (np.dot(state['P1'][j], state['beta1'][start_i1:end_i1]) - \
     np.diag(state['P1'][j])*state['beta1'][start_i1:end_i1])
    b2 = np.dot(state['R2'][j], state['beta_margin2_'][start_i2:end_i2]) -  \
    (np.dot(state['P2'][j], state['beta2'][start_i2:end_i2]) - \
     np.diag(state['P2'][j])*state['beta2'][start_i2:end_i2])

    ## b1_ and b2_ are only for SNPs shared between trait1 and trait2
    b1_ = np.dot(s1*state['R1'][j], state['beta_margin1_'][start_i1:end_i1])[idx1_shared[j]] + \
    np.dot(s2*state['R2'][j], state['beta_margin2_'][start_i2:end_i2])[idx2_shared[j]] - \
    (np.dot(s1*state['P1'][j], state['beta1'][start_i1:end_i1])[idx1_shared[j]] - \
     (np.diag(s1*state['P1'][j])*state['beta1'][start_i1:end_i1])[idx1_shared[j]])- \
    (np.dot(s2*state['P2'][j], state['beta2'][start_i2:end_i2])[idx2_shared[j]]  - \
     (np.diag(s2*state['P2'][j])*state['beta2'][start_i2:end_i2])[idx2_shared[j]] )
    
    b2_ = np.dot(s2*state['R1'][j], state['beta_margin1_'][start_i1:end_i1])[idx1_shared[j]] + \
    np.dot(s3*state['R2'][j], state['beta_margin2_'][start_i2:end_i2])[idx2_shared[j]]  - \
    (np.dot(s2*state['P1'][j], state['beta1'][start_i1:end_i1])[idx1_shared[j]] - \
     (np.diag(s2*state['P1'][j])*state['beta1'][start_i1:end_i1])[idx1_shared[j]])- \
    (np.dot(s3*state['P2'][j], state['beta2'][start_i2:end_i2])[idx2_shared[j]] - \
     (np.diag(s3*state['P2'][j])*state['beta2'][start_i2:end_i2])[idx2_shared[j]] )

    state['b1'][start_i1:end_i1] = b1
    state['b2'][start_i2:end_i2] = b2
    state['b1_'][start_i1:end_i1][idx1_shared[j]]  = b1_
    state['b2_'][start_i2:end_i2][idx2_shared[j]]  = b2_

    
def vectorized_random_choice(prob_matrix, items):
    s = prob_matrix.cumsum(axis=0)
    r = np.random.rand(prob_matrix.shape[1])
    k = (s < r).sum(axis=0)
    k[np.where(k == len(items))] = len(items) - 1
    return items[k]

def sample_assignment(j, idx1_shared, idx2_shared, ld_boundaries1, ld_boundaries2, ref_ld_mat1, ref_ld_mat2, state):
    start_i1 = ld_boundaries1[j][0]; end_i1 = ld_boundaries1[j][1]
    start_i2 = ld_boundaries2[j][0]; end_i2 = ld_boundaries2[j][1]
    m = state['num_clusters_'][2]; N1 = state['N1_']; N2 = state['N2_']
    b1 = state['b1'][start_i1:end_i1].reshape((end_i1-start_i1, 1))
    b2 = state['b2'][start_i2:end_i2].reshape((end_i2-start_i2, 1))
    B1 = state['P1'][j]; B2 = state['P2'][j]
    
    b1_ = state['b1_'][start_i1:end_i1].reshape((end_i1-start_i1, 1))
    b2_ = state['b2_'][start_i2:end_i2].reshape((end_i2-start_i2, 1))
    
    s=state['ovp_pcor']
    s1=N1/(1-s**2);s2=-s*np.sqrt(N1*N2)/(1-s**2);s3=N2/(1-s**2)
    B1_ = s1*state['P1'][j]; B2_ = s2*state['P1'][j]; B3_ = s3*state['P2'][j]


    log_prob_mat = np.zeros((len(idx1_shared[j]), m))
    assignment1 = np.zeros(len(b1))
    assignment2 = np.zeros(len(b2))
    

    idx = range(0 , state['population'][1])
    cluster_var = state['cluster_var_specific'][2][idx]
    pi = np.array(state['pi'][2])[idx]
    C = -.5 * np.log(N1*np.outer(np.diag(B1[idx1_shared[j],:][:,idx1_shared[j]]), cluster_var) + 1) + \
        np.log( pi + 1e-40 )
    a = (N1*b1[idx1_shared[j]])**2 / (2 * np.add.outer(N1 * np.diag(B1[idx1_shared[j],:][:,idx1_shared[j]]),  1.0/cluster_var[1:]) )
    log_prob_mat[:,idx] = np.insert(a, 0, 0, axis=1) + C
    
    idx = range(state['population'][1], state['population'][2])
    cluster_var = state['cluster_var_specific'][2][idx]
    pi = np.array(state['pi'][2])[idx]
    C = -.5 * np.log(N2*np.outer(np.diag(B2[idx2_shared[j],:][:,idx2_shared[j]]), cluster_var) + 1) + \
        np.log( pi + 1e-40 )
    a = (N2*b2[idx2_shared[j]])**2 / (2 * np.add.outer(N2 * np.diag(B2[idx2_shared[j],:][:,idx2_shared[j]]),  1.0/cluster_var) )
    log_prob_mat[:,idx] = a + C
    
    idx = range(state['population'][2], state['population'][3])
    cluster_var1 = state['cluster_var1'][0]
    cluster_var2 = state['cluster_var2'][0]
    cluster_gcov = state['cluster_rho'][0]
    pi = np.array(state['pi'][2])[idx]
    n_snp = len(idx1_shared[j])
    ak1 = np.add.outer(0.5*np.diag(B1_[idx1_shared[j],:][:,idx1_shared[j]]),  .5*cluster_var2/(cluster_var1*cluster_var2-cluster_gcov**2))
    ak2 = np.add.outer(0.5*np.diag(B3_[idx2_shared[j],:][:,idx2_shared[j]]),  .5*cluster_var1/(cluster_var1*cluster_var2-cluster_gcov**2))
    ck = np.add.outer(-np.diag(B2_[idx1_shared[j],:][:,idx1_shared[j]]), cluster_gcov/(cluster_var1*cluster_var2-cluster_gcov**2))
    mu1 = (2*b1_[idx1_shared[j]] + b2_[idx2_shared[j]]*(ck/ak2)) / (4*ak1 - ck**2/ak2)
    mu2 = (2*b2_[idx2_shared[j]] + b1_[idx1_shared[j]]*(ck/ak1)) / (4*ak2 - ck**2/ak1)
    
    C = -.5*np.log(4*ak1*ak2-ck**2) - .5*np.log(cluster_var1*cluster_var2-cluster_gcov**2)  + np.log( pi + 1e-40 )
      
    a = ak1*mu1*mu1 + ak2*mu2*mu2 - ck*mu1*mu2
    log_prob_mat[:,idx] = a + C
    
    logexpsum = special.logsumexp(log_prob_mat, axis=1).reshape((len(idx1_shared[j]), 1))
    prob_mat = np.exp(log_prob_mat - logexpsum)
    
    assignment_shared = vectorized_random_choice(prob_mat.T, np.array(state['cluster_ids_'][2]))
    assignment1[idx1_shared[j]] = assignment_shared
    assignment2[idx2_shared[j]] = assignment_shared

    # trait1 specific variants
    idx_pop1 = np.setdiff1d(np.array(range(len(b1))), idx1_shared[j])
    if (len(idx_pop1) > 0):
        cluster_var = state['cluster_var_specific'][0]
        pi = np.array(state['pi'][0])
        C = -.5 * np.log(N1*np.outer(np.diag(B1[idx_pop1,:][:,idx_pop1]), cluster_var) + 1) + \
                np.log( pi + 1e-40 )
        a = (N1*b1[idx_pop1])**2 / (2 * np.add.outer(N1 * np.diag(B1[idx_pop1,:][:,idx_pop1]),  1.0/cluster_var[1:]) )
        log_prob_mat = np.insert(a, 0, 0, axis=1) + C
        logexpsum = special.logsumexp(log_prob_mat, axis=1).reshape((len(idx_pop1), 1))
        prob_mat = np.exp(log_prob_mat - logexpsum)
        assignment1[idx_pop1] = vectorized_random_choice(prob_mat.T, np.array(state['cluster_ids_'][0]))

    # trait2 specific variants
    idx_pop2 = np.setdiff1d(np.array(range(len(b2))), idx2_shared[j])
    if (len(idx_pop2) > 0):
        cluster_var = state['cluster_var_specific'][1]
        pi = np.array(state['pi'][1])
        C = -.5 * np.log(N2*np.outer(np.diag(B2[idx_pop2,:][:,idx_pop2]), cluster_var) + 1) + \
                np.log( pi + 1e-40 )
        a = (N2*b2[idx_pop2])**2 / (2 * np.add.outer(N2 * np.diag(B2[idx_pop2,:][:,idx_pop2]),  1.0/cluster_var[1:]) )
        log_prob_mat = np.insert(a, 0, 0, axis=1) + C
        logexpsum = special.logsumexp(log_prob_mat, axis=1).reshape((len(idx_pop2), 1))
        prob_mat = np.exp(log_prob_mat - logexpsum)
        assignment2[idx_pop2] = vectorized_random_choice(prob_mat.T, np.array(state['cluster_ids_'][1]))

    return assignment1, assignment2

def sample_pi_pop(state):
    m = np.array([0.0]*4)
    m[0] += state['suffstats'][2][0]
    m[1] += np.sum(list(state['suffstats'][2].values())[1:state['population'][1]])
    m[2] += np.sum(list(state['suffstats'][2].values())[state['population'][1]:state['population'][2]])
    m[3] += np.sum(list(state['suffstats'][2].values())[state['population'][2]:state['population'][3]])
    state['suff_pop'] = m
    state['pi_pop'] = dict(zip(range(0, 4), stats.dirichlet(m+1).rvs()[0]))

def sample_V(state):
    # shared variants
    for j in range(1,4):
        m = len(state['V'][2][j])
        suffstats = np.array(list(state['suffstats'][2].values())[state['population'][j-1]:state['population'][j]])
        a = 1 + suffstats[:-1]
        b = state['alpha'][2][j] + np.cumsum(suffstats[::-1])[:-1][::-1]
        sample_val = stats.beta(a=a, b=b).rvs()
        if 1 in sample_val:
            idx = np.argmax(sample_val == 1)
            sample_val[idx+1:] = 0
            sample_return = dict(zip(range(m-1), sample_val))
            sample_return[m-1] = 0
        else:
            sample_return = dict(zip(range(m-1), sample_val))
            sample_return[m-1] = 1
        state['V'][2][j] = list(sample_return.values())

    # pop1 & pop2 specific variants
    for pop in range(0,2):
        m = len(state['V'][pop])
        suffstats = np.array(list(state['suffstats'][pop].values()))
        a = 1 + suffstats[:-1]
        b = state['alpha'][pop] + np.cumsum(suffstats[::-1])[:-1][::-1]
        sample_val = stats.beta(a=a, b=b).rvs()
        if 1 in sample_val:
            idx = np.argmax(sample_val == 1)
            sample_val[idx+1:] = 0
            sample_return = dict(zip(range(m-1), sample_val))
            sample_return[m-1] = 0
        else:
            sample_return = dict(zip(range(m-1), sample_val))
            sample_return[m-1] = 1
        state['V'][pop] = list(sample_return.values())

def update_p(state):

    # shared variants
    state['pi'][2][0] = state['pi_pop'][0]
    for j in range(1,4):
        m = len(state['V'][2][j])
        V = state['V'][2][j]
        a = np.cumprod(1-np.array(V)[0:(m-2)])*V[1:(m-1)]
        pi = dict(zip(range(1, m), a))
        pi[0] = state['V'][2][j][0]
        pi[m-1] = 1 - np.sum(list(pi.values())[0:(m-1)])

        # last p may be less than 0 due to rounding error
        if pi[m-1] < 0:
            pi[m-1] = 0
        idx = range(state['population'][j-1], state['population'][j])
        state['pi'][2][idx] = np.array(list(pi.values()))*state['pi_pop'][j]

   # pop1 & pop 2 specific variants
    for pop in range(0,2):
        m = len(state['V'][pop])
        V = state['V'][pop]
        a = np.cumprod(1-np.array(V)[0:(m-2)])*V[1:(m-1)]
        pi = dict(zip(range(1, m), a))
        pi[0] = state['V'][pop][0]
        pi[m-1] = 1 - np.sum(list(pi.values())[0:(m-1)])
        if pi[m-1] < 0:
            pi[m-1] = 0
        state['pi'][pop] = np.array(list(pi.values()))

# Sample alpha
def sample_alpha(state):
    # shared variants
    for j in range(1,4):
        m = np.size(np.where( np.array(state['V'][2][j]) != 0)); V = state['V'][2][j]
        a = state['hyperparameters_']['a0'] + m - 1
        b = state['hyperparameters_']['b0'] - np.sum( np.log( 1 - np.array(V[0:(m-1)]) ) )
        state['alpha'][2][j] = stats.gamma(a=a, scale=1.0/b).rvs()

    # pop1 & pop2 specific variants
    for pop in range(0,2):
        m = np.size(np.where( np.array(state['V'][pop]) != 0)); V = state['V'][pop]
        a = state['hyperparameters_']['a0'] + m - 1
        b = state['hyperparameters_']['b0'] - np.sum( np.log( 1 - np.array(V[0:(m-1)]) ) )
        state['alpha'][pop] = stats.gamma(a=a, scale=1.0/b).rvs()

def sample_MVN(mu, cov):
    rv = stats.norm.rvs(size=mu.shape[0])
    C = linalg.cholesky(cov, lower=True)
    return np.dot(C, rv) + mu


def compute_varg(j, state, ld_boundaries1, ld_boundaries2, ref_ld_mat1, ref_ld_mat2):
    start_i1 = ld_boundaries1[j][0]
    end_i1 = ld_boundaries1[j][1]
    start_i2 = ld_boundaries2[j][0]
    end_i2 = ld_boundaries2[j][1]
    ref_ld1 = ref_ld_mat1[j]
    ref_ld2 = ref_ld_mat2[j]
    state['varg1'][j] = np.sum(state['beta1'][start_i1:end_i1] * np.dot(ref_ld1, state['beta1'][start_i1:end_i1]))
    state['varg2'][j] = np.sum(state['beta2'][start_i2:end_i2] * np.dot(ref_ld2, state['beta2'][start_i2:end_i2]))

def sample_beta(j, state, idx1_shared, idx2_shared, ld_boundaries1, ld_boundaries2, ref_ld_mat1, ref_ld_mat2,
               ref_ld_mat_,ID_trait1,ID_trait2):
    M=state['M']
    start_i1 = ld_boundaries1[j][0]
    end_i1 = ld_boundaries1[j][1]
    start_i2 = ld_boundaries2[j][0]
    end_i2 = ld_boundaries2[j][1]
    
    cluster_rho_combine =np.concatenate( ( np.array([0]*len(state['cluster_var_specific'][2])),state['cluster_rho'][0] ))
    rho1 = cluster_rho_combine[state['assignment1'][start_i1:end_i1]]
    rho2 = cluster_rho_combine[state['assignment2'][start_i2:end_i2]]

    N1 = state['N1_']; N2 = state['N2_']; s=state['ovp_pcor']
    s1=N1/(1-s**2);s2=-s*np.sqrt(N1*N2)/(1-s**2);s3=N2/(1-s**2)
    beta_margin1 = state['beta_margin1_'][start_i1:end_i1]; beta_margin2 = state['beta_margin2_'][start_i2:end_i2]

    A1 = state['R1'][j]; B1 = state['P1'][j]
    A2 = state['R2'][j]; B2 = state['P2'][j]
    A1_ = s1*state['R1'][j]; B1_ = s1*state['P1'][j]
    A2_ = s2*state['R'][j]; B2_ = s2*state['P'][j]
    A3_ = s3*state['R2'][j]; B3_ = s3*state['P2'][j]
    

    cluster_var1_combine = np.concatenate((state['cluster_var_specific'][2][:state['population'][1]],np.array([0]*(M[2])),state['cluster_var1'][0] ))
    cluster_var2_combine = np.concatenate((np.array([0]*state['population'][1]), state['cluster_var_specific'][2][state['population'][1]:state['population'][2]],state['cluster_var2'][0] ))
    cluster_var1 = cluster_var1_combine[state['assignment1'][start_i1:end_i1]]
    cluster_var2 = cluster_var2_combine[state['assignment2'][start_i2:end_i2]]

    idx_pop_var1 = np.setdiff1d(np.array(range(end_i1-start_i1)), idx1_shared[j])
    idx_pop_var2 = np.setdiff1d(np.array(range(end_i2-start_i2)), idx2_shared[j])
    cluster_var1[idx_pop_var1] = state['cluster_var_specific'][0][state['assignment1'][start_i1:end_i1][idx_pop_var1]]
    cluster_var2[idx_pop_var2] = state['cluster_var_specific'][1][state['assignment2'][start_i2:end_i2][idx_pop_var2]]

    beta1 = np.zeros(len(beta_margin1)); beta2 = np.zeros(len(beta_margin2))

    # null
    idx1_null = state['assignment1'][start_i1:end_i1] == 0
    idx2_null = state['assignment2'][start_i2:end_i2] == 0

    # trait1 specific
    # only considering beta1 is sufficient
    idx1 = (state['assignment1'][start_i1:end_i1] >= 1) \
            & (state['assignment1'][start_i1:end_i1] < state['population'][1])
    idx1[idx_pop_var1] = state['assignment1'][start_i1:end_i1][idx_pop_var1] != 0

    # trait2 speicifc
    idx2 = (state['assignment2'][start_i2:end_i2] >= state['population'][1]) \
        & (state['assignment2'][start_i2:end_i2] < state['population'][2])
    idx2[idx_pop_var2] = state['assignment2'][start_i2:end_i2][idx_pop_var2] != 0


    # shared with correlation
    idx3_1 = (state['assignment1'][start_i1:end_i1] >= state['population'][2]) \
        & (state['assignment1'][start_i1:end_i1] < state['population'][3])
    idx3_2 = (state['assignment2'][start_i2:end_i2] >= state['population'][2]) \
    & (state['assignment2'][start_i2:end_i2] < state['population'][3])

    idx_pop1 = np.logical_or(idx1, idx3_1)
    idx_pop2 = np.logical_or(idx2, idx3_2)

    ### For both traits, all SNPs are assigned in component 1                                                                       
    if all(idx1_null) and all(idx2_null):
        pass
    ### There are SNPs for trait1 assigned in component 2 and sll SNPs in trait2 are assigned in component 1  
    ### Only SNPs in trait1 are causal.                                                                        
    elif sum(idx1) > 1 and sum(idx_pop2) == 0:
        shrink_ld = B1[idx1,:][:,idx1]
        mat = N1*shrink_ld + np.diag(1.0 / cluster_var1[idx1])
        chol, low = linalg.cho_factor(mat, overwrite_a=False)
        cov_mat = linalg.cho_solve((chol, low), np.eye(chol.shape[0]))
        mu = N1*np.dot(cov_mat, A1[:, idx1].T).dot(beta_margin1)
        beta1[idx1 == 1] = sample_MVN(mu, cov_mat)
    ### Only one SNP in trait1 is causal
    elif sum(idx1) == 1 and sum(idx_pop2) == 0:
        var_k = cluster_var1[idx1]
        const = var_k / (var_k*np.squeeze(B1[idx1,:][:,idx1]) + 1.0/N1)
        bj = state['b1'][start_i1:end_i1][idx1]
        beta1[idx1 == 1] = np.sqrt(const*1.0/N1)*stats.norm.rvs() + const*bj
    ### Only SNPs in trait2 are causal
    elif sum(idx2) > 1 and sum(idx_pop1) == 0:
        shrink_ld = B2[idx2,:][:,idx2]
        mat = N2*shrink_ld + np.diag(1.0 / cluster_var2[idx2])
        chol, low = linalg.cho_factor(mat, overwrite_a=False)
        cov_mat = linalg.cho_solve((chol, low), np.eye(chol.shape[0]))
        mu = N2*np.dot(cov_mat, A2[:, idx2].T).dot(beta_margin2)
        beta2[idx2 == 1] = sample_MVN(mu, cov_mat)
    elif sum(idx2) == 1 and sum(idx_pop1) == 0:
        var_k = cluster_var2[idx2]
        const = var_k / (var_k*np.squeeze(B2[idx2,:][:,idx2]) + 1.0/N2)
        bj = state['b2'][start_i2:end_i2][idx2]
        beta2[idx2 == 1] = np.sqrt(const*1.0/N2)*stats.norm.rvs() + const*bj
    else:
        # two population LD matrix
        #shrink_ld = np.block([[N1*B1[idx_pop1,:][:,idx_pop1], np.zeros((sum(idx_pop1), sum(idx_pop2)))],
        #     [np.zeros((sum(idx_pop2), sum(idx_pop1))), N2*B2[idx_pop2,:][:,idx_pop2]]])
        shrink_ld = np.block([[B1_[idx_pop1,:][:,idx_pop1],B2_[ID_trait1[j],:][:,ID_trait2[j]][idx_pop1,:][:,idx_pop2]],
             [B2_[ID_trait2[j],:][:,ID_trait1[j]][idx_pop2,:][:,idx_pop1], B3_[idx_pop2,:][:,idx_pop2]]])
        # variance covariance matrix for beta
        idx_cor1 = np.where(state['assignment1'][start_i1:end_i1][idx_pop1] >= state['population'][2])[0]
        idx_cor2 = np.where(state['assignment2'][start_i2:end_i2][idx_pop2] >= state['population'][2])[0]

        diag1 = np.diag(cluster_var1[idx_pop1])
        cor1 = np.zeros((sum(idx_pop1), sum(idx_pop2)))
        diag2 = np.diag(cluster_var2[idx_pop2])
        cor2 = np.zeros((sum(idx_pop2), sum(idx_pop1)))

        for i in range(len(idx_cor1)):
            assert rho1[idx_pop1][idx_cor1][i] == rho2[idx_pop2][idx_cor2][i]
            rho = rho1[idx_pop1][idx_cor1][i]
            cor1[idx_cor1[i],idx_cor2[i]] = rho
            cor2[idx_cor2[i],idx_cor1[i]] = rho
        var_mat = np.block([[diag1, cor1],
                    [cor2, diag2]])
        chol1, low1 = linalg.cho_factor(var_mat, overwrite_a=False)
        var_mat_inv = linalg.cho_solve((chol1, low1), np.eye(chol1.shape[0]))

        mat = shrink_ld + var_mat_inv

        chol, low = linalg.cho_factor(mat, overwrite_a=False)
        cov_mat = linalg.cho_solve((chol, low), np.eye(chol.shape[0]))

        A_gamma = np.concatenate([np.dot(A1_[idx_pop1,:], state['beta_margin1_'][start_i1:end_i1])+\
                                  np.dot(A2_[ID_trait1[j],:][:,ID_trait2[j]][idx_pop1,:], state['beta_margin2_'][start_i2:end_i2]),
                                  np.dot(A2_[ID_trait2[j],:][:,ID_trait1[j]][idx_pop2,:], state['beta_margin1_'][start_i1:end_i1])+\
                                  np.dot(A3_[idx_pop2,:], state['beta_margin2_'][start_i2:end_i2])])

        mu = np.dot(cov_mat, A_gamma)
        beta_tmp = sample_MVN(mu, cov_mat)
        beta1[idx_pop1] = beta_tmp[0:sum(idx_pop1)]
        beta2[idx_pop2] = beta_tmp[sum(idx_pop1):]

    state['beta1'][start_i1:end_i1] = beta1
    state['beta2'][start_i2:end_i2] = beta2

def gibbs_stick_break(state,  idx1_shared, idx2_shared, ld_boundaries1, ld_boundaries2, 
                      ref_ld_mat1, ref_ld_mat2, ref_ld_mat_, ID_trait1, ID_trait2, n_threads):
    state['cluster_var_specific'] = sample_sigma2(state)
    sample_gamma(state)
    sample_theta(state)

    cluster=sample_IW(state)
    state['cluster_var1'] = [cluster[0]]
    state['cluster_var2'] = [cluster[1]]
    state['cluster_rho'] = [cluster[2]]

    sample_delta(state)
    sample_lambda(state)

    for j in range(len(ld_boundaries1)):
        calc_b(j, state, ld_boundaries1, ld_boundaries2, idx1_shared, idx2_shared)

    tmp = Parallel(n_jobs=n_threads, require='sharedmem')(delayed(sample_assignment)(j=j, idx1_shared=idx1_shared, idx2_shared=idx2_shared, ld_boundaries1=ld_boundaries1, ld_boundaries2=ld_boundaries2, ref_ld_mat1=ref_ld_mat1, ref_ld_mat2=ref_ld_mat2, state=state) for j in range(len(ld_boundaries1)))



    state['assignment1'] = np.concatenate([tmp[j][0].astype(int) for j in range(len(ld_boundaries1))])
    state['assignment2'] = np.concatenate([tmp[j][1].astype(int) for j in range(len(ld_boundaries1))])

    state['suffstats'] = update_suffstats(state)
    sample_pi_pop(state)

    sample_V(state)
    update_p(state)
    sample_alpha(state)

    for j in range(len(ld_boundaries1)):
        sample_beta(j, state, idx1_shared=idx1_shared, idx2_shared=idx2_shared, ld_boundaries1=ld_boundaries1, 
                    ld_boundaries2=ld_boundaries2, ref_ld_mat1=ref_ld_mat1, ref_ld_mat2=ref_ld_mat2,
                   ref_ld_mat_ = ref_ld_mat_, ID_trait1 = ID_trait1, ID_trait2 = ID_trait2)
        compute_varg(j, state, ld_boundaries1, ld_boundaries2, ref_ld_mat1, ref_ld_mat2)

    state['h2_1'] = np.sum(state['varg1'])
    state['h2_2'] = np.sum(state['varg2'])



