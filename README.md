# LDSC (LD SCore) `v1.0.2`

`ldsc` is a command line tool for estimating heritability and genetic correlation from GWAS summary statistics. `ldsc` also computes LD Scores.

## Python 3 Compatibility

This repository is a Python 3 compatible fork of the original [LDSC project](https://github.com/bulik/ldsc). The codebase has been updated to work with modern Python 3.8+ environments while maintaining the same functionality.

Key improvements include:
- Compatible with Python 3.8 and newer
- Updated dependency requirements
- Fixed syntax for print statements, file handling, and other Python 2 specific code
- Modern pandas interface usage

## Getting Started

In order to download `ldsc`, you should clone this repository via the commands:
```  
git clone https://github.com/buutrg/ldsc_py3.git
cd ldsc_py3
```

In order to install the Python dependencies, you will need the [Anaconda](https://store.continuum.io/cshop/anaconda/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) Python distribution and package manager. After installing Anaconda/Miniconda, run the following commands to create an environment with LDSC's dependencies:

```
conda env create --file environment.yml
conda activate ldsc
```

This will create a Python 3.8 environment with all the necessary dependencies.

Once the above has completed, you can run:

```
./ldsc.py -h
./munge_sumstats.py -h
```
to print a list of all command-line options. If these commands fail with an error, then something has gone wrong during the installation process. 

Short tutorials describing the four basic functions of `ldsc` (estimating LD Scores, h2 and partitioned h2, genetic correlation, the LD Score regression intercept) can be found in the [wiki](https://github.com/bulik/ldsc/wiki). If you would like to run the tests, please see the wiki.

## Updating LDSC

You can update to the newest version of `ldsc` using `git`. First, navigate to your `ldsc_py3/` directory (e.g., `cd ldsc_py3`), then run:
```
git pull
```
If `ldsc` is up to date, you will see: 
```
Already up-to-date.
```
Otherwise, you will see `git` output telling you which files were changed. If you have modified the `ldsc` source code, `git pull` may fail with an error such as `error: Your local changes to the following files would be overwritten by merge:`. 

In case the Python dependencies have changed, you can update the LDSC environment with:

```
conda env update --file environment.yml
```

## Python Version Compatibility

This fork has been specifically created to work with Python 3.8+. The original [LDSC project](https://github.com/bulik/ldsc) is compatible with Python 2.7, which is now deprecated.

Major changes in this Python 3 version include:
- Updated dependency versions in environment.yml and requirements.txt
- Modernized print() function calls
- Replaced xrange() with range()
- Updated file handling with correct text/binary mode
- Fixed pandas DataFrame and Series methods to use newer interfaces
- Explicit encoding specifications for file operations
- Proper handling of dictionary views vs. lists

All functionality remains the same as the original project with these implementation updates.

## Where Can I Get LD Scores?

You can download [European](https://data.broadinstitute.org/alkesgroup/LDSCORE/eur_w_ld_chr.tar.bz2) and [East Asian LD Scores](https://data.broadinstitute.org/alkesgroup/LDSCORE/eas_ldscores.tar.bz2) from 1000 Genomes [here](https://data.broadinstitute.org/alkesgroup/LDSCORE/). These LD Scores are suitable for basic LD Score analyses (the LD Score regression intercept, heritability, genetic correlation, cross-sex genetic correlation). You can download partitioned LD Scores for partitioned heritability estimation [here](http://data.broadinstitute.org/alkesgroup/LDSCORE/).

## Support

Before contacting us, please try the following:

1. The [wiki](https://github.com/bulik/ldsc/wiki) has tutorials on [estimating LD Score](https://github.com/bulik/ldsc/wiki/LD-Score-Estimation-Tutorial), [heritability, genetic correlation and the LD Score regression intercept](https://github.com/bulik/ldsc/wiki/Heritability-and-Genetic-Correlation) and [partitioned heritability](https://github.com/bulik/ldsc/wiki/Partitioned-Heritability).
2. Common issues are described in the [FAQ](https://github.com/bulik/ldsc/wiki/FAQ)
3. The methods are described in the papers (citations below)

For issues specific to this Python 3 port, please open an issue in this repository.

For general LDSC questions, you can get in touch with the original authors via the [google group](https://groups.google.com/forum/?hl=en#!forum/ldsc_users).

Issues with LD Hub? Email ld-hub@bristol.ac.uk

## Citation

If you use the software or the LD Score regression intercept, please cite:

[Bulik-Sullivan, et al. LD Score Regression Distinguishes Confounding from Polygenicity in Genome-Wide Association Studies.
Nature Genetics, 2015.](http://www.nature.com/ng/journal/vaop/ncurrent/full/ng.3211.html)

For genetic correlation, please also cite:

[Bulik-Sullivan, B., et al. An Atlas of Genetic Correlations across Human Diseases and Traits. Nature Genetics, 2015.](https://www.nature.com/articles/ng.3406) Preprint available on bioRxiv doi: http://dx.doi.org/10.1101/014498

For partitioned heritability, please also cite:

[Finucane, HK, et al. Partitioning heritability by functional annotation using genome-wide association summary statistics. Nature Genetics, 2015.](https://www.nature.com/articles/ng.3404) Preprint available on bioRxiv doi: http://dx.doi.org/10.1101/014241

For stratified heritability using continuous annotation, please also cite:

[Gazal, S, et al. Linkage disequilibrium–dependent architecture of human complex traits shows action of negative selection. Nature Genetics, 2017.](https://www.nature.com/articles/ng.3954) 

If you find the fact that LD Score regression approximates HE regression to be conceptually useful, please cite:

Bulik-Sullivan, Brendan. Relationship between LD Score and Haseman-Elston, bioRxiv doi: http://dx.doi.org/10.1101/018283

For LD Hub, please cite:

[Zheng, et al. LD Hub: a centralized database and web interface to perform LD score regression that maximizes the potential of summary level GWAS data for SNP heritability and genetic correlation analysis. Bioinformatics (2016)](https://doi.org/10.1093/bioinformatics/btw613)

## License

This project is licensed under GNU GPL v3.

## Authors

Original LDSC:
- Brendan Bulik-Sullivan (Broad Institute of MIT and Harvard)
- Hilary Finucane (MIT Department of Mathematics)

Python 3 port:
- Buu Truong
