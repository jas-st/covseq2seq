"""
Convert amino acid positions to nt positions and vice versa.

@Alice_Wittig @JakubBartoszewicz @MelaniaNowicka
"""
from functools import lru_cache
import math
try:
    from shared.residue_converter.data_loader import get_gene_pos, get_reference
except ModuleNotFoundError:  # pragma: no cover
    from data_loader import get_gene_pos, get_reference


def aa2na_gene(aa):
    """
    Convert amino acid position to gene position.

    Args:
        aa: int, amino acid position

    Returns:
        (int, int), output gene na position (codon start, codon end)
    """
    factor = (aa - 1)*3
    return 1+factor, 1+factor+2


def aa2na_genome(aa, gene="S"):
    # input aa
    # output whole genome na positon (codon start, codon end)
    factor = (aa - 1)*3
    gene_regions = get_gene_pos()[gene]
    gene_start = gene_regions[-1][1]
    gene_length = gene_regions[-1][2] - gene_start + 1
    if aa > (gene_length / 3):
        raise ValueError(f"error AA position {aa} is behind the end of the gene {gene}.")
    return gene_start+factor, gene_start+factor+2


@lru_cache(maxsize=128)
def na2aa_gene(na):
    # input gene na position
    # output aa
    return math.floor((na-1)/3)+1


def na2aa_genome(na, gene="S"):
    # input whole genome position
    # output aa
    gene_regions = get_gene_pos()[gene]
    gene_start = gene_regions[-1][1]
    gene_stop = gene_regions[-1][2]
    if na < gene_start or na > gene_stop:
        raise ValueError(f"Position of NT {na} outside gene {gene}.")
    return math.floor((na-gene_start)/3)+1

@lru_cache(maxsize=128)
def na_gene_to_genome(na, gene="S"):
    # input spike na position
    # output whole genome na positon
    gene_start = get_gene_pos()[gene][-1][1]
    gene_length = get_gene_pos()[gene][-1][2] - gene_start + 1
    if na < 1 or na > gene_length:
        raise ValueError(f"Gene position must be in range {1}-{gene_length} for gene {gene}.")
    return na - 1 + gene_start


def na_genome_to_gene(na, gene="S"):
    # input whole genome na positon
    # output gene na position
    gene_regions = get_gene_pos()[gene]
    gene_start = gene_regions[-1][1]
    gene_stop = gene_regions[-1][2]
    if na < 1 or na > gene_stop:
        raise ValueError(f"Whole-genome position must be in range {gene_start}-{gene_stop} for gene {gene}.")
    return na + 1 - gene_start


def na_genome_to_codon_coordinates(na, gene="S"):
    # input whole genome na positon
    # output: (position of the first nucleotide of the codon, position of input nucleotide relative to codon start)
    nt_gene_position = na_genome_to_gene(na, gene)
    # get the position of the nucleotide variant within a codon (based on its position in the gene). 0-indexed
    nt_position_in_codon = (nt_gene_position - 1) % 3
    # get the position of the first nucleotide of this codon
    # based on the variant's genomic position and its position within the codon. 1-indexed
    codon_start = na - nt_position_in_codon
    return codon_start, nt_position_in_codon


def na2aa_msa_gene(na_str):
    # input na position in msa
    # output aa position in msa
    na_str_split = na_str.split(".")
    if len(na_str_split) > 1:
        na_ref, na_insert = int(na_str_split[0]), int(na_str_split[1])
        return ".".join([str(na2aa_gene(na_ref)), str(na2aa_gene(na_insert))])
    else:
        return str(na2aa_gene(int(na_str_split[0])))
