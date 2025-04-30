## PleioSDPR
PleioSDPR is a statistical method for multi-trait prediction of complex traits. It integrates GWAS summary statistics and LD matrices from two complex traits to compute polygenic scores.

## Installation

You can download PleioSDPR by running

```
git clone https://github.com/Harmony-Chi//PleioSDPR.git
```

We recommend you to run PleioSDPR in the [Anaconda](https://docs.anaconda.com/anaconda/install/index.html).

```
conda create --name PleioSDPR python=3.9
conda activate PleioSDPR
conda install pandas==1.1.5
conda install numpy==1.24.4
conda install joblib
conda install scikit-learn
conda install plinkio
```

## Input 

### Reference LD

The reference LD matrices are based on 1000 Genome Hapmap3 SNPs and can be downloaded from [link](https://yaleedu-my.sharepoint.com/:f:/g/personal/chi_zhang_cz354_yale_edu/EuB9GFTYinFPkF0pWSie8ZABom82mlfSvyspb_ZITNSgbA?e=Umkbgk)

### Summary Statistics 

Same as SDPR, the summary statistics should have at least following columns with the same name, where SNP is the marker name, A1 is the effect allele, A2 is the alternative allele, Z is the Z score for the association statistics, and N is the sample size. 

```
SNP     A1      A2      Z       N
rs737657        A       G       -2.044      252156
rs7086391       T       C       -2.257      248425
rs1983865       T       C       3.652    253135
...
```

## Running PleioSDPR

An example command to run the test data on chromosome 1 when there is no sample overlap: 

```
python ./PleioSDPR/PleioSDPR.py --ss1 trait1.txt --ss2 trait2.txt --N1 n1 --N2 n2 --load_ld ./ref/ --chr 1 --threads 3 --out trait1_trait2_chr1
```

- ss1 (required): Path to the summary statistics of trait1.
- ss2 (required): Path to the summary statistics of trait2.
- N1 (required): Sample size of the summary statistics of trait1.
- N2 (required): Sample size of the summary statistics of trait2.
- load_ld (required): Path to the reference LD directory.
- chr (required): Chromosome.
- out (required): Path to the output file containing estimated effect sizes.
- threads (optional): number of threads to use. Default is 1.


An example command to run the test data on chromosome 1 when there is sample overlap: 

```
python ./PleioSDPR_Ns/PleioSDPR.py --ss1 trait1.txt --ss2 trait2_Ns.txt --N1 n1 --N2 n2 --Ns ns --load_ld ./ref/ --chr 1 --threads 3 --out trait1_trait2_chr1_Ns
```
An additional required options is:

- Ns (required): Overlap ample size of between the summary statistics of trait1 and trait2.

For real data analysis, it is recommended to run each PleioSDPR on each chromosome in parallel, and using 3 threads for each chromsome.  

## Output 

There are two output files corresponding to the adjusted effect sizes for trait1 (e.g. trait1_trait2_chr1_1.txt) and trait2 (e.g. trait1_trait2_chr1_2.txt). you can combine all the chromosomes together and then can use [PLINK](https://www.cog-genomics.org/plink/1.9/score) to derive the PRS. A linear combination of the PGSs of the two traits can be linear combined if there is also a validataion dataset.

## Citation


