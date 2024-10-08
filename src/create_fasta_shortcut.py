import data_loader as dl
import data_handler2 as dh

path = "C:\\Users\\zhasmi00\\Downloads\\test\\ba286_test.txt"
with open(path, "r") as file:
    input_muts = file.readline().replace(" ", "").split(",")
input_muts = [x.upper() for x in input_muts]
input_muts.sort(key=lambda x: int(x[1:-1]))

print(input_muts)
ref_seq = [x for x in dl.REFSEQ]


full_seq = dh.get_full_seq(ref_seq, input_muts)

with open("C:\\Users\\zhasmi00\\Downloads\\test_fasta.txt", "w") as file:
    fastastring = ">%s\n%s\n" % ("seq", full_seq)
    file.write(fastastring)