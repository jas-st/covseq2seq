from Bio import SeqIO
import tarfile

path = "C:\\Users\\zhasmi00\\Downloads\\xbb1_16.xz"

# tar = tarfile.open(path)
# tar.extractall()
# tar.close()


with open("lineage_XBB_1_16_11.fasta") as ffile:
    record = next(SeqIO.parse(ffile, "fasta"))
    print(record.seq)
