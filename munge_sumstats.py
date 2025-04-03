#!/usr/bin/env python
# from __future__ import division  # Not needed in Python 3
import pandas as pd
import numpy as np
import os
import sys
import traceback
import gzip
import bz2
import argparse
from scipy.stats import chi2
from ldscore import sumstats
from ldsc import MASTHEAD, Logger, sec_to_str
import time
np.seterr(invalid='ignore')

try:
    x = pd.DataFrame({'A': [1, 2, 3]})
    x.sort_values(by='A')
except AttributeError:
    raise ImportError('LDSC requires pandas version >= 0.17.0')

null_values = {

    'LOG_ODDS': 0,
    'BETA': 0,
    'OR': 1,
    'Z': 0
}

default_cnames = {
    # RS NUMBER
    'SNP': 'SNP',
    'MARKERNAME': 'SNP',
    'SNPID': 'SNP',
    'RS': 'SNP',
    'RSID': 'SNP',
    'RS_NUMBER': 'SNP',
    'RS_NUMBERS': 'SNP',
    # NUMBER OF STUDIES
    'NSTUDY': 'NSTUDY',
    'N_STUDY': 'NSTUDY',
    'NSTUDIES': 'NSTUDY',
    'N_STUDIES': 'NSTUDY',
    # P-VALUE
    'P': 'P',
    'PVALUE': 'P',
    'P_VALUE':  'P',
    'PVAL': 'P',
    'P_VAL': 'P',
    'GC_PVALUE': 'P',
    # ALLELE 1
    'A1': 'A1',
    'ALLELE1': 'A1',
    'ALLELE_1': 'A1',
    'EFFECT_ALLELE': 'A1',
    'REFERENCE_ALLELE': 'A1',
    'INC_ALLELE': 'A1',
    'EA': 'A1',
    # ALLELE 2
    'A2': 'A2',
    'ALLELE2': 'A2',
    'ALLELE_2': 'A2',
    'OTHER_ALLELE': 'A2',
    'NON_EFFECT_ALLELE': 'A2',
    'DEC_ALLELE': 'A2',
    'NEA': 'A2',
    # N
    'N': 'N',
    'NCASE': 'N_CAS',
    'CASES_N': 'N_CAS',
    'N_CASE': 'N_CAS',
    'N_CASES': 'N_CAS',
    'N_CONTROLS': 'N_CON',
    'N_CAS': 'N_CAS',
    'N_CON': 'N_CON',
    'N_CASE': 'N_CAS',
    'NCONTROL': 'N_CON',
    'CONTROLS_N': 'N_CON',
    'N_CONTROL': 'N_CON',
    'WEIGHT': 'N',  # metal does this. possibly risky.
    # SIGNED STATISTICS
    'ZSCORE': 'Z',
    'Z-SCORE': 'Z',
    'GC_ZSCORE': 'Z',
    'Z': 'Z',
    'OR': 'OR',
    'B': 'BETA',
    'BETA': 'BETA',
    'LOG_ODDS': 'LOG_ODDS',
    'EFFECTS': 'BETA',
    'EFFECT': 'BETA',
    'SIGNED_SUMSTAT': 'SIGNED_SUMSTAT',
    # INFO
    'INFO': 'INFO',
    # MAF
    'EAF': 'FRQ',
    'FRQ': 'FRQ',
    'MAF': 'FRQ',
    'FRQ_U': 'FRQ',
    'F_U': 'FRQ',
}

describe_cname = {
    'SNP': 'Variant ID (e.g., rs number)',
    'P': 'p-Value',
    'A1': 'Allele 1, interpreted as ref allele for signed sumstat.',
    'A2': 'Allele 2, interpreted as non-ref allele for signed sumstat.',
    'N': 'Sample size',
    'N_CAS': 'Number of cases',
    'N_CON': 'Number of controls',
    'Z': 'Z-score (0 --> no effect; above 0 --> A1 is trait/risk increasing)',
    'OR': 'Odds ratio (1 --> no effect; above 1 --> A1 is risk increasing)',
    'BETA': '[linear/logistic] regression coefficient (0 --> no effect; above 0 --> A1 is trait/risk increasing)',
    'LOG_ODDS': 'Log odds ratio (0 --> no effect; above 0 --> A1 is risk increasing)',
    'INFO': 'INFO score (imputation quality; higher --> better imputation)',
    'FRQ': 'Allele frequency',
    'SIGNED_SUMSTAT': 'Directional summary statistic as specified by --signed-sumstats.',
    'NSTUDY': 'Number of studies in which the SNP was genotyped.'
}

numeric_cols = ['P', 'N', 'N_CAS', 'N_CON', 'Z', 'OR', 'BETA', 'LOG_ODDS', 'INFO', 'FRQ', 'SIGNED_SUMSTAT', 'NSTUDY']

def read_header(fh):
    '''Read the first line of a file and returns a list with the column names.'''
    (openfunc, compression) = get_compression(fh)
    # Use text mode for reading with encoding='utf-8' for Python 3 compatibility
    if compression == 'gzip':
        return [x.rstrip('\n') for x in openfunc(fh, 'rt', encoding='utf-8').readline().split()]
    elif compression == 'bz2':
        return [x.rstrip('\n') for x in openfunc(fh, 'rt', encoding='utf-8').readline().split()]
    else:
        return [x.rstrip('\n') for x in openfunc(fh, 'r', encoding='utf-8').readline().split()]


def get_cname_map(flag, default, ignore):
    '''
    Figure out which column names to use.

    Priority is
    (1) ignore everything in ignore
    (2) use everything in flags that is not in ignore
    (3) use everything in default that is not in ignore or in flags

    The keys of flag are cleaned. The entries of ignore are not cleaned. The keys of defualt
    are cleaned. But all equality is modulo clean_header().

    '''
    clean_ignore = [clean_header(x) for x in ignore]
    cname_map = {x: flag[x] for x in flag if x not in clean_ignore}
    cname_map.update(
        {x: default[x] for x in default if x not in clean_ignore + list(flag.keys())})  # Convert keys() to list for Python 3
    return cname_map


def get_compression(fh):
    '''
    Read filename suffixes and figure out whether it is gzipped,bzip2'ed or not compressed
    '''
    if fh.endswith('gz'):
        compression = 'gzip'
        openfunc = gzip.open
    elif fh.endswith('bz2'):
        compression = 'bz2'
        openfunc = bz2.BZ2File
    else:
        openfunc = open
        compression = None

    return openfunc, compression


def clean_header(header):
    '''
    For cleaning file headers.
    - convert to uppercase
    - replace dashes '-' with underscores '_'
    - replace dots '.' (as in R) with underscores '_'
    - remove newlines ('\n')
    '''
    return header.upper().replace('-', '_').replace('.', '_').replace('\n', '')


def filter_pvals(P, log, args):
    '''Remove out-of-bounds P-values'''
    ii = (P > 0) & (P <= 1)
    bad_p = (~ii).sum()
    if bad_p > 0:
        msg = 'WARNING: {N} SNPs had P outside of (0,1]. The P column may be mislabeled.'
        log.log(msg.format(N=bad_p))

    return ii


def filter_info(info, log, args):
    '''Remove INFO < args.info_min (default 0.9) and complain about out-of-bounds INFO.'''
    if type(info) is pd.Series:  # one INFO column
        jj = ((info > 2.0) | (info < 0)) & info.notnull()
        ii = info >= args.info_min
    elif type(info) is pd.DataFrame:  # several INFO columns
        jj = (((info > 2.0) & info.notnull()).any(axis=1) | (
            (info < 0) & info.notnull()).any(axis=1))
        ii = (info.sum(axis=1) >= args.info_min * (len(info.columns)))
    else:
        raise ValueError('Expected pd.DataFrame or pd.Series.')

    bad_info = jj.sum()
    if bad_info > 0:
        msg = 'WARNING: {N} SNPs had INFO outside of [0,2]. The INFO column may be mislabeled.'
        log.log(msg.format(N=bad_info))

    return ii


def filter_frq(frq, log, args):
    '''Filter on MAF.'''
    if type(frq) is pd.Series:
        ii = (frq >= args.maf_min) & (frq <= (1 - args.maf_min))
    else:
        raise ValueError('Expected pd.Series.')

    return ii


def filter_alleles(a):
    '''Remove alleles that do not describe strand-unambiguous SNPs'''
    return a.isin(['A', 'T', 'G', 'C'])


def parse_dat(dat_gen, convert_colname, merge_alleles, log, args):
    '''Parse and filter a dataset with the given column configuration.'''
    dat = next(dat_gen)
    if len(dat) == 0:
        raise ValueError('Input file contains no SNPs')

    dat = dat.dropna(how='all').reset_index(drop=True)
    if 'SNP' in convert_colname:
        dat.rename(columns={convert_colname['SNP']: 'SNP'}, inplace=True, errors='ignore')
        if 'SNP' not in dat.columns:
            raise ValueError('--snp column not found in data.')

        log.log('Writing results for {N} SNPs.'.format(N=len(dat)))
        if args.snp:
            # dat = dat[dat.SNP.isin(args.snp)].reset_index(drop=True)
            dat = pd.merge(pd.DataFrame({'SNP': args.snp}), dat, how='left', on='SNP')
            log.log('Keeping {N} SNPs in --snp.'.format(N=len(dat)))
    else:
        if args.snp:
            raise ValueError('Cannot find SNPs to keep: --snp column not found in data.')

        if 'SNP' not in dat.columns:
            dat['SNP'] = range(len(dat))
            msg = 'WARNING: no SNP column found. Constructed dummy SNP column as row numbers.'
            log.log(msg)

    # Remove bad P-values
    if 'P' in convert_colname:
        dat.rename(columns={convert_colname['P']: 'P'}, inplace=True, errors='ignore')
        if 'P' not in dat.columns:
            if args.merge_alleles or args.daner or args.keep_maf:
                raise ValueError('P column not found in data.')
        else:
            dat = dat[filter_pvals(dat.P, log, args)].reset_index(drop=True)

    # Remove bad ALLELES
    if merge_alleles:
        dat.rename(columns={convert_colname['A1']: 'A1'}, inplace=True, errors='ignore')
        dat.rename(columns={convert_colname['A2']: 'A2'}, inplace=True, errors='ignore')
        if 'A1' not in dat.columns or 'A2' not in dat.columns:
            raise ValueError('A1 or A2 column not found in data.')

        dat = dat[filter_alleles(dat.A1) & filter_alleles(dat.A2)].reset_index(drop=True)

    # Filter on INFO (imputation quality)
    if args.info_min is not None and 'INFO' in convert_colname:
        dat.rename(columns={convert_colname['INFO']: 'INFO'}, inplace=True, errors='ignore')
        info_exists = False
        if 'INFO' in dat.columns:
            info_exists = True
        else:
            # try INFO1 / INFO2 for multi-study data
            pattern = 'INFO[0-9]+'
            info_cols = [c for c in dat.columns if c.match(pattern)]
            if len(info_cols) > 0:
                dat.rename(columns={c: c.replace('INFO', 'INFO_') for c in info_cols},
                           inplace=True)
                info_cols = ['INFO_' + str(i) for i in range(len(info_cols))]
                dat.rename(columns={c: info_cols[i] for i, c in enumerate(info_cols)},
                           inplace=True)
                dat['INFO'] = dat[info_cols].mean(axis=1)
                info_exists = True

        if info_exists:
            dat = dat[filter_info(dat.INFO, log, args)].reset_index(drop=True)

    # Filter on MAF
    if args.maf_min is not None and 'FRQ' in convert_colname:
        dat.rename(columns={convert_colname['FRQ']: 'FRQ'}, inplace=True, errors='ignore')
        if 'FRQ' in dat.columns:
            dat = dat[filter_frq(dat.FRQ, log, args)].reset_index(drop=True)

    return dat


def process_n(dat, args, log):
    '''Determine sample size from args.n* and/or dat.N* columns.'''
    if all(i in dat.columns for i in ['N_CAS', 'N_CON']):
        N = dat.N_CAS + dat.N_CON
        P = dat.N_CAS / N
        dat['N'] = N
    elif 'N' in dat.columns:
        N = dat.N
        P = None
    else:
        if args.N:
            try:
                N = int(args.N)
                P = None
                dat['N'] = N
                log.log('Using N = {N}.'.format(N=N))
            except ValueError:
                raise ValueError('N must be an integer.')
        elif args.N_cas and args.N_con:
            N = int(args.N_cas) + int(args.N_con)
            P = float(args.N_cas) / float(N)
            dat['N'] = N
            dat['N_CAS'] = float(args.N_cas)
            dat['N_CON'] = float(args.N_con)
            log.log('Using N = {N}, N_cas = {N_cas}, N_con = {N_con}, P = {P}.'.format(
                N=N, N_cas=args.N_cas, N_con=args.N_con, P=round(P, 4)))

        else:
            raise ValueError('Could not determine N.')

    return N, P


def p_to_z(P, N):
    '''Convert P-value and N to standardized beta.'''
    return np.sqrt(chi2.isf(P, 1))


def check_median(x, expected_median, tolerance, name):
    '''Check that median(x) is close to expected_median.'''
    m = np.median(x)
    if (expected_median != 0 and abs(m / expected_median - 1) > tolerance) or\
            (expected_median == 0 and abs(m - expected_median) > tolerance):
        msg = 'WARNING: median value of {V} is {M}, which differs from expected {E} by more than {T}. This column may be mislabeled.'
        raise ValueError(msg.format(V=name, M=round(m, 4), E=expected_median, T=tolerance))


def parse_flag_cnames(log, args):
    '''Parse flags that specify how to interpret columns.'''
    if args.nstudy:
        args.nstudy = clean_header(args.nstudy)
    if args.snp:
        args.snp = [clean_header(x) for x in args.snp.split(',')]
    if args.N_col:
        args.N_col = clean_header(args.N_col)
    if args.N_cas_col:
        args.N_cas_col = clean_header(args.N_cas_col)
    if args.N_con_col:
        args.N_con_col = clean_header(args.N_con_col)
    if args.a1:
        args.a1 = clean_header(args.a1)
    if args.a2:
        args.a2 = clean_header(args.a2)
    if args.p:
        args.p = clean_header(args.p)
    if args.frq:
        args.frq = clean_header(args.frq)
    if args.info:
        args.info = clean_header(args.info)
    if args.info_list:
        args.info_list = [clean_header(x) for x in args.info_list.split(',')]
    if args.signed_sumstats:
        try:
            parsed = args.signed_sumstats.split(',')
            if len(parsed) == 1:
                args.signed_sumstats = parsed[0]
                args.signed_sumstats_col = 'SIGNED_SUMSTAT'
            elif len(parsed) == 2:
                args.signed_sumstats, args.signed_sumstats_col = parsed
                args.signed_sumstats_col = clean_header(args.signed_sumstats_col)
            else:
                raise ValueError(
                    'Argument --signed-sumstats must have the form SIGN_COLUMN or SIGN_COLUMN,SIGNED_SUMSTAT_COLNAME.')

        except ValueError as e:
            log.log(str(e))
            raise

    null_value = None
    if args.signed_sumstats_col:
        null_value = args.signed_sumstats_col

    # if we have both z and beta/or columns, delete one
    if args.z and (args.beta or args.or_col):
        raise ValueError(
            'At most one of --z, --beta, --or is allowed.')

    if args.z:
        args.z = clean_header(args.z)
        if null_value:
            null_value += ',Z'
        else:
            null_value = 'Z'
    elif args.beta:
        args.beta = clean_header(args.beta)
        if null_value:
            null_value += ',BETA'
        else:
            null_value = 'BETA'
    elif args.or_col:
        args.or_col = clean_header(args.or_col)
        if null_value:
            null_value += ',OR'
        else:
            null_value = 'OR'

    return null_value


def allele_merge(dat, alleles, log):
    '''
    Merge with reference panel alleles.
    '''
    
    if 'A1' not in dat.columns or 'A2' not in dat.columns:
        raise ValueError('A1/A2 not found in --sumstats.')

    if 'A1' not in alleles.columns or 'A2' not in alleles.columns:
        raise ValueError('A1/A2 not found in --merge-alleles.')

    # standardize and check
    for df in (dat, alleles):
        df.A1 = df.A1.str.upper()
        df.A2 = df.A2.str.upper()
        if not df.A1.isin(set(['A', 'C', 'G', 'T'])).all() or not df.A2.isin(set(['A', 'C', 'G', 'T'])).all():
            raise ValueError('A1/A2 must be one of A,C,G,T.')

    log.log('Merging with reference panel SNPs.')
    merged = pd.merge(dat, alleles, how='inner', on='SNP')

    # flip strand if necessary
    merged['FLIP'] = False
    strand_ambiguous = ((merged.A1_x == merged.A1_y) & (merged.A2_x == merged.A2_y)) | \
                       ((merged.A1_x == merged.A2_y) & (merged.A2_x == merged.A1_y))
    flip = ~strand_ambiguous & (((merged.A1_x == complement(merged.A1_y)) & (merged.A2_x == complement(merged.A2_y))) | \
                              ((merged.A1_x == complement(merged.A2_y)) & (merged.A2_x == complement(merged.A1_y))))
    merged.loc[flip, 'FLIP'] = True
    log.log('Flipped {n} SNPs to match reference panel.'.format(n=flip.sum()))

    # drop non-matching alleles
    mismatch = ~strand_ambiguous & ~flip
    merged = merged[~mismatch]
    log.log('Dropped {n} SNPs due to allele mismatch with reference.'.format(n=mismatch.sum()))

    # check for flipped allele order
    merged['FLIPPED'] = False
    allele_order = (merged.A1_x == merged.A2_y) & (merged.A2_x == merged.A1_y)
    merged.loc[allele_order, 'FLIPPED'] = True
    log.log('Flipped allele order for {n} SNPs.'.format(n=allele_order.sum()))

    # update values
    if 'Z' in merged.columns:
        merged.loc[merged.FLIPPED, 'Z'] = -merged.loc[merged.FLIPPED, 'Z']
    if 'BETA' in merged.columns:
        merged.loc[merged.FLIPPED, 'BETA'] = -merged.loc[merged.FLIPPED, 'BETA']
    if 'OR' in merged.columns:
        merged.loc[merged.FLIPPED, 'OR'] = 1/merged.loc[merged.FLIPPED, 'OR']

    # select relevant columns
    out_columns = [c for c in dat.columns if c not in ['A1', 'A2']]
    out = merged[out_columns + ['A1_y', 'A2_y']].rename(columns={'A1_y': 'A1', 'A2_y': 'A2'})
    return out

def complement(x):
    '''Return complementary alleles.'''
    comp = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    if isinstance(x, str):
        return comp.get(x, x)
    else:  # assume it's a Series
        return x.map(comp.get)

def munge_sumstats(args, p=True):
    '''Parse summary statistics file.'''
    if args.sumstats is None:
        raise ValueError('Error: --sumstats is required.')

    if args.out is None:
        raise ValueError('Error: --out is required.')

    start_time = time.time()
    log = Logger(args.out + '.log')
    if p:
        log.log(MASTHEAD)
        try:
            log.log('Command: {}'.format(' '.join(sys.argv)))
        except:
            pass

    # Get compression type
    compression = 'gzip' if args.sumstats.endswith('.gz') else \
                 'bz2' if args.sumstats.endswith('.bz2') else None

    # Read input options
    if args.a1 is None:
        args.a1 = 'A1'
    if args.a2 is None:
        args.a2 = 'A2'
    if args.p is None:
        args.p = 'P'
    if args.snp is None:
        args.snp = 'SNP'

    # Read header
    header_df = pd.read_csv(args.sumstats, sep=r'\s+', nrows=1, compression=compression)
    header = list(header_df.columns)
    log.log('Found header: {}'.format(header))

    # Map column names
    cname_map = {}
    for k, v in default_cnames.items():
        if k in header:
            cname_map[k] = v
            log.log('Found column {k}, interpreting as {v}'.format(k=k, v=v))

    # Override with user-specified column names
    if args.snp:
        cname_map[args.snp] = 'SNP'
    if args.a1:
        cname_map[args.a1] = 'A1'
    if args.a2:
        cname_map[args.a2] = 'A2'
    if args.p:
        cname_map[args.p] = 'P'
    if args.N_col:
        cname_map[args.N_col] = 'N'
    if args.N_cas_col:
        cname_map[args.N_cas_col] = 'N_CAS'
    if args.N_con_col:
        cname_map[args.N_con_col] = 'N_CON'
    if args.frq:
        cname_map[args.frq] = 'FRQ'
    if args.info:
        cname_map[args.info] = 'INFO'
    if args.z:
        cname_map[args.z] = 'Z'
    if args.beta:
        cname_map[args.beta] = 'BETA'
    if args.or_col:
        cname_map[args.or_col] = 'OR'

    # Read data
    usecols = [c for c in header if c in cname_map]
    
    # Include SE column in usecols if we need to calculate Z
    if 'BETA' in [cname_map.get(c, '') for c in header] and 'SE' in header:
        if 'SE' not in usecols:
            usecols.append('SE')
    
    log.log('Reading columns: {}'.format(usecols))
    
    try:
        dat = pd.read_csv(args.sumstats, sep=r'\s+', usecols=usecols, compression=compression)
        log.log('Read {N} SNPs from --sumstats file.'.format(N=len(dat)))
    except Exception as e:
        log.log('Error reading --sumstats file: {}'.format(e))
        raise

    # Rename columns
    dat.rename(columns=cname_map, inplace=True)

    # Filter bad values
    if 'P' in dat.columns:
        bad_p = (dat.P <= 0) | (dat.P > 1)
        dat = dat[~bad_p]
        log.log('Removed {N} SNPs with invalid P values.'.format(N=sum(bad_p)))

    # Filter INFO
    if 'INFO' in dat.columns:
        dat = dat[dat.INFO >= args.info_min]
        log.log('Removed SNPs with INFO < {I}.'.format(I=args.info_min))

    # Filter MAF
    if 'FRQ' in dat.columns and args.maf_min is not None:
        dat = dat[(dat.FRQ >= args.maf_min) & (dat.FRQ <= (1 - args.maf_min))]
        log.log('Removed SNPs with MAF < {M}.'.format(M=args.maf_min))

    # Merge alleles if necessary
    if args.merge_alleles:
        log.log('Reading --merge-alleles from {F}'.format(F=args.merge_alleles))
        merge_alleles = pd.read_csv(args.merge_alleles, sep=r'\s+')
        dat = allele_merge(dat, merge_alleles, log)

    # Check sample size
    n, p = process_n(dat, args, log)

    # Calculate Z if needed
    if 'Z' not in dat.columns:
        if 'P' in dat.columns:
            # Silently compute Z from P-values without logging
            sign = np.sign(dat['BETA']) if 'BETA' in dat.columns else 1
            dat['Z'] = sign * p_to_z(dat['P'], n)
        elif 'BETA' in dat.columns and 'SE' in dat.columns:
            # Silently compute Z from BETA/SE without logging
            dat['Z'] = dat['BETA'] / dat['SE']
        else:
            log.log('Warning: Cannot compute Z score - missing required columns')
    
    # Make sure all required columns exist
    required_columns = ['SNP', 'A1', 'A2', 'N', 'Z']
    
    # Check which columns are missing
    missing_columns = [col for col in required_columns if col not in dat.columns]
    for col in missing_columns:
        log.log(f'Warning: {col} column not found in data')
    
    # Add any missing columns with placeholder values
    for col in missing_columns:
        dat[col] = np.nan
    
    # Select only the columns we want in the exact order specified
    output_dat = dat[required_columns]
    
    # Output
    out_fname = args.out + '.sumstats.gz'
    log.log('Writing compressed output to {F}'.format(F=out_fname))
    output_dat.to_csv(out_fname, sep='\t', index=False, na_rep='NA', compression='gzip')
    log.log('Wrote {N} SNPs.'.format(N=len(output_dat)))

    time_elapsed = round(time.time() - start_time, 2)
    log.log('Time elapsed: {T}'.format(T=sec_to_str(time_elapsed)))
    
    return dat

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sumstats', default=None, type=str,
                        help="Input summary statistics file.")
    parser.add_argument('--out', default=None, type=str,
                        help="Output filename prefix.")
    parser.add_argument('--N', default=None, type=int,
                        help="Sample size.")
    parser.add_argument('--N-cas', default=None, type=int,
                        help="Number of cases.")
    parser.add_argument('--N-con', default=None, type=int,
                        help="Number of controls.")
    parser.add_argument('--no-alleles', default=False, action="store_true",
                        help="Don't require alleles. Useful for e.g., continuous phenotypes.")
    parser.add_argument('--maf-min', default=0.01, type=float,
                        help="Minimum MAF.")
    parser.add_argument('--info-min', default=0.9, type=float,
                        help="Minimum INFO score.")
    parser.add_argument('--daner', default=False, action="store_true",
                        help="Use Daner format.")
    parser.add_argument('--keep-maf', default=False, action="store_true",
                        help="Keep the MAF column (instead of dropping it).")
    parser.add_argument('--merge-alleles', default=None, type=str,
                        help="Merge alleles with file (useful for sorting out strand issues)")
    parser.add_argument('--n-min', default=None, type=float,
                        help='Minimum N (sample size). Default: filter out bottom 10%% of SNPs.')
    parser.add_argument('--chunksize', default=5e6, type=int,
                        help='Chunksize.')
    
    # Optional column names
    parser.add_argument('--snp', default=None, type=str,
                        help='Name of SNP column (if not a name that ldsc understands). NB: case insensitive.')
    parser.add_argument('--N-col', default=None, type=str,
                        help='Name of N column (if not a name that ldsc understands). NB: case insensitive.')
    parser.add_argument('--N-cas-col', default=None, type=str,
                        help='Name of N column (if not a name that ldsc understands). NB: case insensitive.')
    parser.add_argument('--N-con-col', default=None, type=str,
                        help='Name of N column (if not a name that ldsc understands). NB: case insensitive.')
    parser.add_argument('--a1', default=None, type=str,
                        help='Name of A1 column (if not a name that ldsc understands). NB: case insensitive.')
    parser.add_argument('--a2', default=None, type=str,
                        help='Name of A2 column (if not a name that ldsc understands). NB: case insensitive.')
    parser.add_argument('--p', default=None, type=str,
                        help='Name of p-value column (if not a name that ldsc understands). NB: case insensitive.')
    parser.add_argument('--frq', default=None, type=str,
                        help='Name of FRQ or MAF column (if not a name that ldsc understands). NB: case insensitive.')
    parser.add_argument('--signed-sumstats', default=None, type=str,
                        help='Name of signed sumstat column, comma null value (e.g., Z,0 or OR,1). NB: case insensitive.')
    parser.add_argument('--info', default=None, type=str,
                        help='Name of INFO column (if not a name that ldsc understands). NB: case insensitive.')
    parser.add_argument('--info-list', default=None, type=str,
                        help='Comma-separated list of INFO columns. Will filter on the mean. NB: case insensitive.')
    parser.add_argument('--nstudy', default=None, type=str,
                        help='Name of NSTUDY column (if not a name that ldsc understands). NB: case insensitive.')
    parser.add_argument('--nstudy-min', default=None, type=int,
                        help='Minimum # of studies. Default: don\'t filter on nstudy.')
    parser.add_argument('--ignore', default=None, type=str,
                        help='Comma-separated list of column names to ignore.')
    parser.add_argument('--a1-inc', default=False, action='store_true',
                        help='A1 is the increasing allele.')
    parser.add_argument('--z', default=None, type=str,
                        help='Name of Z column. Must be a name that ldsc understands.')
    parser.add_argument('--beta', default=None, type=str,
                        help='Name of beta column. Must be a name that ldsc understands.')
    parser.add_argument('--or-col', default=None, type=str,
                        help='Name of OR column. Must be a name that ldsc understands.')
    
    args = parser.parse_args()
    
    munge_sumstats(args, p=True)
