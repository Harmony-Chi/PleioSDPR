#!/usr/bin/env python
import numpy as np
from scipy import stats
try:
    import pickle as pickle
except:
    import cPickle as pickle
import gzip
import gibbs, ld
import argparse
import sys, util
import pandas as pd


def PleioSDPR_gibbs(beta_margin1, beta_margin2, N1, N2,  idx1_shared, idx2_shared, ld_boundaries1, ld_boundaries2, 
ref_ld_mat1, ref_ld_mat2,ref_ld_mat_, ID_trait1 , ID_trait2 , mcmc_samples,     burn,  n_threads, chr):
    M = [1, 1000,1000,1000]
    trace = {'sigma1_2':[],'sigma2_2':[],'gc':[], 'num_cluster':[], 'beta1':np.zeros(shape=(mcmc_samples, len(beta_margin1))),
        'beta2':np.zeros(shape=(mcmc_samples, len(beta_margin2))),
        'suffstats':[], 'h2_1':[], 'h2_2':[]}
    state = gibbs.initial_state(data1=beta_margin1, data2=beta_margin2, idx1_shared=idx1_shared, idx2_shared=idx2_shared,
            ld_boundaries1=ld_boundaries1, ld_boundaries2=ld_boundaries2, M=M, N1=N1, N2=N2, a0k=.5, b0k=.5)
    state['suffstats'] = gibbs.update_suffstats(state)

    state['a'] = 0.1; state['c'] = 1

    state['R1'] = [np.linalg.solve(ref_ld_mat1[j] + state['a']*np.identity(ref_ld_mat1[j].shape[0]), ref_ld_mat1[j]) for j in range(len(ld_boundaries1))]
    state['P1'] = [np.dot(ref_ld_mat1[j], state['R1'][j]) for j in range(len(ld_boundaries1))]
    state['R2'] = [np.linalg.solve(ref_ld_mat2[j] + state['a']*np.identity(ref_ld_mat2[j].shape[0]), ref_ld_mat2[j]) for j in range(len(ld_boundaries2))]
    state['P2'] = [np.dot(ref_ld_mat2[j], state['R2'][j]) for j in range(len(ld_boundaries2))]
    state['R'] = [np.linalg.solve(ref_ld_mat_[j] + state['a']*np.identity(ref_ld_mat_[j].shape[0]), ref_ld_mat_[j]) for j in range(len(ld_boundaries2))]
    state['P'] = [np.dot(ref_ld_mat_[j], state['R'][j]) for j in range(len(ld_boundaries2))]


    for i in range(mcmc_samples):
        gibbs.gibbs_stick_break(state,  idx1_shared=idx1_shared, idx2_shared=idx2_shared, ld_boundaries1=ld_boundaries1, ld_boundaries2=ld_boundaries2, ref_ld_mat1=ref_ld_mat1, ref_ld_mat2=ref_ld_mat2,ref_ld_mat_ = ref_ld_mat_, ID_trait1 = ID_trait1, ID_trait2 = ID_trait2,  n_threads=n_threads)

        if (i > burn):
            trace['h2_1'].append(state['h2_1'])
            trace['h2_2'].append(state['h2_2'])

        trace['beta1'][i,] = state['beta1']
        trace['beta2'][i,] = state['beta2']

        if (state['h2_1'] == 0  and state['h2_2'] == 0):
            state = gibbs.initial_state(data1=beta_margin1, data2=beta_margin2, idx1_shared=idx1_shared, idx2_shared=idx2_shared, 
                    ld_boundaries1=ld_boundaries1, ld_boundaries2=ld_boundaries2, M=M, N1=N1, N2=N2, a0k=.5, b0k=.5)
            state['suffstats'] = gibbs.update_suffstats(state)
            state['a'] = 0.1; state['c'] = 1
            state['R1'] = [np.linalg.solve(ref_ld_mat1[j] + state['a']*np.identity(ref_ld_mat1[j].shape[0]), ref_ld_mat1[j]) for j in range(len(ld_boundaries1))]
            state['P1'] = [np.dot(ref_ld_mat1[j], state['R1'][j]) for j in range(len(ld_boundaries1))]

            state['R2'] = [np.linalg.solve(ref_ld_mat2[j] + state['a']*np.identity(ref_ld_mat2[j].shape[0]), ref_ld_mat2[j]) for j in range(len(ld_boundaries2))]
            state['P2'] = [np.dot(ref_ld_mat2[j], state['R2'][j]) for j in range(len(ld_boundaries2))]
            state['R'] = [np.linalg.solve(ref_ld_mat_[j] + state['a']*np.identity(ref_ld_mat_[j].shape[0]), ref_ld_mat_[j]) for j in range(len(ld_boundaries2))]
            state['P'] = [np.dot(ref_ld_mat_[j], state['R'][j]) for j in range(len(ld_boundaries2))]



        util.progressBar(value=i+1, endvalue=mcmc_samples)

    # calculate posterior average
    poster_mean1 = np.mean(trace['beta1'][burn:mcmc_samples], axis=0)
    poster_mean2 = np.mean(trace['beta2'][burn:mcmc_samples], axis=0)
    

    return poster_mean1, poster_mean2

def pipeline(args):
    
    N1 = args.N1; N2 = args.N2

    print('Load summary statistics from {}'.format(args.ss1))
    ss1 = pd.read_table(args.ss1, sep=None)
    print('Load summary statistics from {}'.format(args.ss2))
    ss2 = pd.read_table(args.ss2, sep=None)

    ref_ld_mat1 = []; ref_ld_mat2 = []; ref_ld_mat_ =[]
    ID_trait1 = [];ID_trait2 = []
    ld_boundaries1 = []; ld_boundaries2 = []
    A1_1 = []; A1_2 = []; SNP1 = []; SNP2 = []
    beta_margin1 = []; beta_margin2 = []
    idx1_shared = []; idx2_shared = []
    left1 = 0; left2 = 0

    f = gzip.open(args.load_ld + '/chr_' + str(args.chr) +'.gz', 'r')
    try:
        ld_dict = pickle.load(f)
    except:
        f.seek(0)
        ld_dict = pickle.load(f, encoding='latin1')
    f.close()
   
    snps = ld_dict[0]; a1 = ld_dict[1]; a2 = ld_dict[2]
    ref_boundary = ld_dict[3]; ref_ld_mat = ld_dict[4]

    ref = pd.DataFrame({'SNP':snps, 'A1':a1, 'A2':a2})
    tmp_ss1 = pd.merge(ref, ss1, on="SNP", how="left")
    tmp_ss2 = pd.merge(ref, ss2, on="SNP", how="left")

    for i in range(len(ref_boundary)):
        tmp_blk_ss1 = tmp_ss1.iloc[ref_boundary[i][0]:ref_boundary[i][1]]
        tmp_blk_ss2 = tmp_ss2.iloc[ref_boundary[i][0]:ref_boundary[i][1]]
        tmp_beta1 = tmp_ss1.iloc[ref_boundary[i][0]:ref_boundary[i][1]]['Z'] / np.sqrt(tmp_ss1.iloc[ref_boundary[i][0]:ref_boundary[i][1]]['N'])
        tmp_beta2 = tmp_ss2.iloc[ref_boundary[i][0]:ref_boundary[i][1]]['Z'] / np.sqrt(tmp_ss2.iloc[ref_boundary[i][0]:ref_boundary[i][1]]['N'] )
        idx1_ss1 = np.logical_and(tmp_blk_ss1['A1_x'] == tmp_blk_ss1['A1_y'], tmp_blk_ss1['A2_x'] == tmp_blk_ss1['A2_y'])
        idx2_ss1 = np.logical_and(tmp_blk_ss1['A1_x'] == tmp_blk_ss1['A2_y'], 
		tmp_blk_ss1['A2_x'] == tmp_blk_ss1['A1_y'])
        tmp_beta1[idx2_ss1] = -tmp_beta1[idx2_ss1] 
        idx1_ss2 = np.logical_and(tmp_blk_ss2['A1_x'] == tmp_blk_ss2['A1_y'], 
		tmp_blk_ss2['A2_x'] == tmp_blk_ss2['A2_y'])
        idx2_ss2 = np.logical_and(tmp_blk_ss2['A1_x'] == tmp_blk_ss2['A2_y'], 
		tmp_blk_ss2['A2_x'] == tmp_blk_ss2['A1_y'])
        tmp_beta2[idx2_ss2] = -tmp_beta2[idx2_ss2] 
        idx1 = np.logical_or(idx1_ss1, idx2_ss1)
        idx2 = np.logical_or(idx1_ss2, idx2_ss2)
	
        idx = np.logical_or(idx1, idx2)
        idx_ = np.logical_and(idx1, idx2)
        if np.sum(idx) == 0:
            continue
	
        if np.sum(idx1) != 0 and np.sum(idx2) != 0 and np.sum(idx2[idx1]) != 0:
            ref_ld_mat1.append(ref_ld_mat[i][idx1,:][:,idx1])
            ld_boundaries1.append([left1, left1+np.sum(idx1)])
            beta_margin1.extend(list(tmp_beta1[idx1])) 
            SNP1.extend(list(tmp_blk_ss1[idx1].SNP))
            A1_1.extend(list(tmp_blk_ss1[idx1].A1_x))
            idx1_shared.append(np.where(idx2[idx1])[0])
            left1 += np.sum(idx1)
            ref_ld_mat2.append(ref_ld_mat[i][idx2,:][:,idx2])
            ld_boundaries2.append([left2, left2+np.sum(idx2)])
            beta_margin2.extend(list(tmp_beta2[idx2]))
            SNP2.extend(list(tmp_blk_ss2[idx2].SNP))
            A1_2.extend(list(tmp_blk_ss2[idx2].A1_x))
            idx2_shared.append(np.where(idx1[idx2])[0])
            left2 += np.sum(idx2)
            ref_ld_mat_.append(ref_ld_mat[i])
            ID_trait1.append(idx1)
            ID_trait2.append(idx2)
            shared_snp = list(set(tmp_blk_ss1[idx1].SNP) & set(tmp_blk_ss2[idx2].SNP))
 
        elif np.sum(idx1) != 0:
            ref_ld_mat1[-1] = np.block([[ref_ld_mat1[-1], np.zeros((ref_ld_mat1[-1].shape[0], np.sum(idx1)))], [np.zeros((np.sum(idx1), ref_ld_mat1[-1].shape[0])), ref_ld_mat[i][idx1,:][:,idx1]]])
            ld_boundaries1[-1][1] += np.sum(idx1)
            beta_margin1.extend(list(tmp_beta1[idx1])) 
            SNP1.extend(list(tmp_blk_ss1[idx1].SNP))
            A1_1.extend(list(tmp_blk_ss1[idx1].A1_x))
            left1 += np.sum(idx1)
        
            ref_ld_mat_[-1] = np.block([[ref_ld_mat_[-1],
                                        np.zeros((ref_ld_mat_[-1].shape[0],ref_ld_mat[i].shape[0]))],
                                      [np.zeros((ref_ld_mat[i].shape[0],ref_ld_mat_[-1].shape[0])),
                                      ref_ld_mat_[i]]])
            ID_trait1[-1] = np.concatenate([ID_trait1[-1] ,idx1])
            ID_trait2[-1] = np.concatenate([ID_trait2[-1] ,idx2])
            


            ref_ld_mat2[-1] = np.block([[ref_ld_mat2[-1], 
                                         np.zeros((ref_ld_mat2[-1].shape[0], np.sum(idx2)))], 
                                        [np.zeros((np.sum(idx2), ref_ld_mat2[-1].shape[0])), 
                                         ref_ld_mat[i][idx2,:][:,idx2]]])
            ld_boundaries2[-1][1] += np.sum(idx2)
            beta_margin2.extend(list(tmp_beta2[idx2]))
            SNP2.extend(list(tmp_blk_ss2[idx2].SNP))
            A1_2.extend(list(tmp_blk_ss2[idx2].A1_x))
            left2 += np.sum(idx2)
	
        elif np.sum(idx2) != 0:
            ref_ld_mat2[-1] = np.block([[ref_ld_mat2[-1], np.zeros((ref_ld_mat2[-1].shape[0], np.sum(idx2)))], [np.zeros((np.sum(idx2), ref_ld_mat2[-1].shape[0])), ref_ld_mat[i][idx2,:][:,idx2]]])
            ld_boundaries2[-1][1] += np.sum(idx2)
            beta_margin2.extend(list(tmp_beta2[idx2]))  
            SNP2.extend(list(tmp_blk_ss2[idx2].SNP))
            A1_2.extend(list(tmp_blk_ss2[idx2].A1_x))
            left2 += np.sum(idx2)
            ref_ld_mat_[-1] = np.block([[ref_ld_mat_[-1],
                                        np.zeros((ref_ld_mat_[-1].shape[0],ref_ld_mat[i].shape[0]))],
                                      [np.zeros((ref_ld_mat[i].shape[0],ref_ld_mat_[-1].shape[0])),
                                      ref_ld_mat_[i]]])
            ID_trait1[-1] = np.concatenate([ID_trait1[-1] ,idx1])
            ID_trait2[-1] = np.concatenate([ID_trait2[-1] ,idx2])



    print('Start MCMC ...')
    res1, res2 = PleioSDPR_gibbs(beta_margin1=np.array(beta_margin1)/args.c1, beta_margin2=np.array(beta_margin2)/args.c2, 
                             N1=N1, N2=N2,  idx1_shared=idx1_shared, idx2_shared=idx2_shared, 
                             ld_boundaries1=ld_boundaries1, ld_boundaries2=ld_boundaries2, 
                             ref_ld_mat1=ref_ld_mat1, ref_ld_mat2=ref_ld_mat2, ref_ld_mat_=ref_ld_mat_, ID_trait1 = ID_trait1, ID_trait2 = ID_trait2, mcmc_samples=args.mcmc_samples, 
                             burn=args.burn, n_threads=args.threads, chr=args.chr)

    print('Done!\nWrite output to {}'.format(args.out+'.txt'))

    out1 = pd.DataFrame({'SNP':SNP1, 'A1':A1_1, 'post_beta':res1})
    out1.to_csv(args.out+'_1.txt', columns=['SNP', 'A1', 'post_beta'], sep="\t", index=False)
    out2 = pd.DataFrame({'SNP':SNP2, 'A1':A1_2, 'post_beta':res2})
    out2.to_csv(args.out+'_2.txt', columns=['SNP', 'A1', 'post_beta'], sep="\t", index=False)


parser = argparse.ArgumentParser(prog='PleioSDPR',
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                description="Version 0.0.1 Test Only")

parser.add_argument('--ss1', type=str, required=True,
                        help='Path to cleaned summary statistics 1. e.g. /home/tutor/myss.txt')

parser.add_argument('--ss2', type=str, required=True,
                        help='Path to cleaned summary statistics 2. e.g. /home/tutor/myss.txt')

parser.add_argument('--N1', type=int, default=None, required=True,
                        help='Number of individuals in summary statistic sile 1.')

parser.add_argument('--N2', type=int, default=None, required=True,
                        help='Number of individuals in summary statistic sile 2.')

parser.add_argument('--chr', type=int, default=None, required=True,
	                        help='Chromosome.')

parser.add_argument('--c1', type=float, default=1.0,
	                                help='C1.')

parser.add_argument('--c2', type=float, default=1.0,
	                                help='C2.')


parser.add_argument('--threads', type=int, default=1, 
                        help='Number of Threads used.')

parser.add_argument('--seed', type=int, 
                        help='Specify the seed for numpy random number generation.')

parser.add_argument('--mcmc_samples', type=int, default=1500,
                        help='Specify the total number of iterations in MCMC.')

parser.add_argument('--burn', type=int, default=200,
                        help='Specify the total number of iterations to be discarded before \
                        Markov Chain approached the stationary distribution.')

parser.add_argument('--save_ld', type=str, default=None,
                        help='Prefix of the location to save calculated LD Reference file \
                        in pickled and gzipped format.')

parser.add_argument('--load_ld', type=str, default=None,
                        help='Prefix of the location to load calculated LD Reference file \
                        in pickled and gzipped format.')


parser.add_argument('--out', type=str, required=True,
                        help='Prefix of the location for the output tab deliminated .txt file.')

def main():
    pipeline(parser.parse_args())

if __name__ == '__main__':
    main()
