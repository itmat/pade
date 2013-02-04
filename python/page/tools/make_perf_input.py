import sys

import argparse

def read_input(filename):
    with open(filename) as f:
        f.next()    
        rows = []

        for line in f:
            line = line.rstrip()
            row = line.split("\t")
            row = row[1:] + row[1:9]
            rows.append("\t".join(row))
    
        return rows


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--features', '-n', type=int, help="Number of features")
    parser.add_argument('filename', help="Base input file")

    args = parser.parse_args()
    headers = [ "id" ]
    for i in range(24):
        headers.append("sample_{0:02d}".format(i))
    rows = read_input(args.filename)

    print "\t".join(headers)
    counter = 0
    while counter < args.features:
        row = rows[counter % len(rows)]
        counter += 1
        print "exon_" + str(counter) + "\t" + row

if __name__ == '__main__':
    main()
