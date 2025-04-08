#!/usr/bin/env python3

import click
import snpio

@click.group()
def cli():
    """
    Command-line interface for genomic analysis tools.
    """
    pass


@cli.command()
@click.option('--vcf', type=click.Path(exists=True), required=True, help='Path to the VCF file')
@click.option('--popmap', type=click.Path(exists=True), required=True, help='Path to the population map file')
@click.option('--prop_pc_var', type=float, default=None, required=False, help='Proportion of variance for PCs')
@click.option('--max_n_clust', type=int, default=20, required=False, help='Maximum number of clusters')
@click.option('--n_pca_min', type=int, default=10, required=False, help='Minimum number of PCs')
@click.option('--n_pca_max', type=int, default=200, required=False, help='Maximum number of PCs')
@click.option('--n_pca_interval', type=int, default=20, required=False, help='Interval for number of PCs')
@click.option('--n_rep', type=int, default=10, required=False, help='Number of repetitions (folds for cross-validation)')
def test(vcf, popmap, prop_pc_var, max_n_clust, n_pca_min, n_pca_max, n_pca_interval, n_rep):
    """
    Perform unsupervised discriminant analysis of principal components 

    This command takes a VCF file and a population map file as input, and performs
    UDAPC analysis with the specified parameters.
    """

    # Load the genotype data from a VCF file
    gd = snpio.VCFReader(
        filename=vcf,
        popmapfile=popmap,
        force_popmap=True,
        verbose=True,
        plot_format="png",
        plot_fontsize=20,
        plot_dpi=300,
        prefix="snpio_example"
    )
    print(gd)

    pass


# @cli.command()
# @click.option('--vcf', type=click.Path(exists=True), required=True, help='Path to the VCF file')
# @click.option('--popmap', type=click.Path(exists=True), required=True, help='Path to the population map file')
# @click.option('--n_pc', type=int, help='Number of principal components')
# @click.option('--n_da', type=int, help='Number of discriminant functions')
# @click.option('--prop_pc', type=float, default=0.9, help='Proportion of variance for PCs')
# @click.option('--prop_da', type=float, default=0.9, help='Proportion of variance for DFs')
# @click.option('--cv_k', type=int, help='K for k-fold cross-validation')
# def dapc(vcf, popmap, n_pc, n_da, prop_pc, prop_da, cv_k):
#     """
#     Perform discriminant analysis of principal components (DAPC).

#     This command takes a VCF file and a population map file as input, and performs
#     DAPC analysis with the specified parameters.
#     """

#     if cv_k:
#         # If cv_k is provided, override n_pc, n_da, prop_pc, and prop_da
#         n_pc = None
#         n_da = None
#         prop_pc = None
#         prop_da = None
#         # Add your cross-validation logic here using cv_k

#     # Load the VCF and population map files using snpio
#     # (replace with actual snpio loading code)
#     pass


if __name__ == '__main__':
    cli()