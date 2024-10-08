import sys
import gzip
import csv

from Bio import SeqIO
from . import utils

import RNA

import RNA
# A dictionary with all types of loops distinguished in the
# unpaired probability computations of the ViennaRNA Package
looptypes = { RNA.EXT_LOOP : "external",
              RNA.HP_LOOP : "hairpin",
              RNA.INT_LOOP : "internal",
              RNA.MB_LOOP : "multibranch"
}

def accessibility(sequence, footprints = 30, windowsize = 200, L = 150, m6A_sites = None):
    """
    Compute unpaired probabilities for a set of (or single) footprint(s)
    using a sliding-window local RNA secondary structure prediction.

    The function returns a dictionary of unpaired probabilities for
    each footprint. Here, the keys are the individual footprint sizes
    as specified in the input argument 'footprints'. Each of the
    corresponding values is a dictionary again, where the keys are the
    individual loop types the footprint may reside in and the values are
    lists of probabilities where the i-th value is the probability of
    a footprint starting at position i.
    """

    def up_callback(v, v_size, i, maxsize, what, data):
        """
        Store accessibility data callback function
        """
        # only process unpaired probabilities
        if what & RNA.PROBS_WINDOW_UP:
            # mask variable 'what' such that it assumes
            # one of the vaues in 'looptypes'
            what = what & ~RNA.PROBS_WINDOW_UP
            #if i % 100 == 0:
            #    print(i)
            for fp in data.keys():
                # we store data for footprints starting
                # at position i, but 'probs_window()' yields
                # data ending at position i. So we need to
                # compute actual start position of the footprint
                start = i - fp + 1
                if start > 0:
                    dat = data[fp][what][start] = v[fp]

    # store footprints as list of footprint sizes
    fps = footprints if type(footprints) is list else [ footprints ]
    fps = [ int(fp) for fp in fps ]

    # create data structure (dict of dicts of lists) where we will store
    # the computed accessibilities. After filling the data structure, we
    # can obtain the resulting unpaired probabilities for each footprint
    # of size 'fp', starting at position 'i' and residing in loop type
    # 'lt' as:
    # p = data[fp][lt][i]
    data = { k : { lt : [ 0 for i in range(len(sequence) - k + 2) ] for lt in looptypes } for k in fps }

    # create model details and set windowsize and maximum base pair span
    md = RNA.md()
    md.max_bp_span = L
    md.window_size = windowsize

    # create fold_compound for sliding-window computations
    fc = RNA.fold_compound(sequence.upper(), md, RNA.OPTION_WINDOW)

    if m6A_sites:
        fc.sc_m6A(m6A_sites)

    # compute sliding-window probabilities
    fc.probs_window(max(fps),
                    RNA.PROBS_WINDOW_UP | RNA.PROBS_WINDOW_UP_SPLIT,
                    up_callback,
                    data)

    return data


def analyze_transcript_diff(transcript_file,
                            intersect_file,
                            output_file,
                            footprints = [5],
                            windowsize = 200,
                            L = 150,
                            max_transcripts = 0):

    if not transcript_file or \
       not intersect_file or \
       not output_file:
        return
    
    m6A_pos = utils.get_local_m6A_sites(intersect_file)
    transcripts_analyzed = 0

    with gzip.open(output_file, 'wt', newline='') as f:
        spamwriter = csv.writer(f, delimiter = ',', quoting=csv.QUOTE_MINIMAL)

        # write a header into the file
        spamwriter.writerow(['transcript','length','m6A_sites','looptype','footprint','data'])

        # go through corresponding FASTA file and process each transcript
        with open(transcript_file) as handle:
            map_errors = 0
            for i, record in enumerate(SeqIO.parse(handle, "fasta")):

                identifier = str(record.id)
                sequence = str(record.seq)

                sys.stdout.write(f'\r{i}. processing \"{identifier}\" (l = {len(sequence)}nt)' + ' ' * 20)
                sys.stdout.flush()
                # check whether we have any m6A data for this transcript identifier
                if identifier in m6A_pos:
                    actual_sites = []
                    # check whether all m6A positions make any sense, i.e. map to an actual A in the sequence
                    for m6A_site in m6A_pos[identifier]:
                        if m6A_site < 0:
                            print("wtf", identifier, m6A_site)
                        elif (m6A_site <= len(sequence)) and (sequence[m6A_site - 1] == "A" or sequence[m6A_site - 1] == "a"):
                            actual_sites.append(m6A_site)

                    if len(actual_sites) == 0:
                        print("mapping failed")
                        map_errors += 1
                        continue
                    elif len(actual_sites) != len(m6A_pos[identifier]):
                        print(f"mapped just {len(actual_sites)} of {len(m6A_pos[identifier])} sites")

                    sys.stdout.write(f'\r{i}. processing \"{identifier}\" (l = {len(sequence)}nt, m6A_sites = {len(actual_sites)})' + ' ' * 20)
                    sys.stdout.flush()

                    # run the predictions
                    data_unmod = accessibility(sequence, footprints, windowsize = windowsize, L = L)
                    data = accessibility(sequence, footprints, windowsize = windowsize, L = L, m6A_sites = actual_sites)
                                                                                      
                    
                    # compute differences between unmodified and modified sequences
                    # and store them into the csv file
                    for lt in looptypes:
                        for u in footprints:
                            dd = [identifier, len(sequence), len(actual_sites), lt, u]
                            dd += [v for v in map(lambda pair: (pair[0] - pair[1])/pair[1] if pair[1] != 0 else 0, zip(data[u][lt], data_unmod[u][lt]))]
                            spamwriter.writerow(dd)

                    transcripts_analyzed += 1
                    if (max_transcripts > 0) and (transcripts_analyzed >= max_transcripts):
                        break

            if map_errors:
                print(f'{map_errors} data sets with map errors')

def m6a_analyze_transcript_diff(transcript_file,
                            intersect_file,
                            output_file,
                            footprints = [5],
                            windowsize = 200,
                            L = 150,
                            max_transcripts = 0):

    if not transcript_file or \
       not intersect_file or \
       not output_file:
        return
    
    m6A_pos = utils.get_local_m6A_sites(intersect_file)
    transcripts_analyzed = 0

    with gzip.open(output_file, 'wt', newline='') as f:
        spamwriter = csv.writer(f, delimiter = ',', quoting=csv.QUOTE_MINIMAL)

        # write a header into the file
        spamwriter.writerow(['transcript','length','m6A_sites','looptype','footprint','data'])

        # go through corresponding FASTA file and process each transcript
        with open(transcript_file) as handle:
            map_errors = 0
            for i, record in enumerate(SeqIO.parse(handle, "fasta")):

                identifier = str(record.id)
                sequence = str(record.seq)

                sys.stdout.write(f'\r{i}. processing \"{identifier}\" (l = {len(sequence)}nt)' + ' ' * 20)
                sys.stdout.flush()
                # check whether we have any m6A data for this transcript identifier
                if identifier in m6A_pos:
                    actual_sites = []
                    # check whether all m6A positions make any sense, i.e. map to an actual A in the sequence
                    for m6A_site in m6A_pos[identifier]:
                        if m6A_site < 0:
                            print("wtf", identifier, m6A_site)
                        elif (m6A_site <= len(sequence)) and (sequence[m6A_site - 1] == "A" or sequence[m6A_site - 1] == "a"):
                            actual_sites.append(m6A_site)

                    if len(actual_sites) == 0:
                        print("mapping failed")
                        map_errors += 1
                        continue
                    elif len(actual_sites) != len(m6A_pos[identifier]):
                        print(f"mapped just {len(actual_sites)} of {len(m6A_pos[identifier])} sites")

                    sys.stdout.write(f'\r{i}. processing \"{identifier}\" (l = {len(sequence)}nt, m6A_sites = {len(actual_sites)})' + ' ' * 20)
                    sys.stdout.flush()

                    # run the predictions
                    data_unmod = accessibility(sequence, footprints, windowsize = windowsize, L = L)
                    data = accessibility(sequence, footprints, windowsize = windowsize, L = L, m6A_sites = actual_sites)
                    
                    new_data_unmod = { k : { lt : [ ] for lt in looptypes } for k in footprints }
                    new_data = { k : { lt : [ ] for lt in looptypes } for k in footprints }
                    
                    for site in actual_sites:             
                        for u in footprints:
                            for lt in looptypes:
                                if len(data_unmod[u][lt]) in [range((site-100),(site+101))]:
                                    new_data_unmod[u][lt].extend(data_unmod[u][lt][(site-100):])
                                    new_data[u][lt].extend(data[u][lt][(site-100):])
                                elif (site-100)<0:
                                    new_data_unmod[u][lt].extend(data_unmod[u][lt][:(site+101)])
                                    new_data[u][lt].extend(data[u][lt][:(site+101)])
                                else:
                                    new_data_unmod[u][lt].extend(data_unmod[u][lt][(site-100):(site+101)])
                                    new_data[u][lt].extend(data[u][lt][(site-100):(site+101)])
                                                                               
                    
                    # compute differences between unmodified and modified sequences
                    # and store them into the csv file
                    for lt in looptypes:
                        for u in footprints:
                            dd = [identifier, len(sequence), len(actual_sites), lt, u]
                            dd += [v for v in map(lambda pair: (pair[0] - pair[1])/pair[1] if pair[1] != 0 else 0, zip(new_data[u][lt], new_data_unmod[u][lt]))]
                            spamwriter.writerow(dd)

                    transcripts_analyzed += 1
                    if (max_transcripts > 0) and (transcripts_analyzed >= max_transcripts):
                        break

            if map_errors:
                print(f'{map_errors} data sets with map errors')