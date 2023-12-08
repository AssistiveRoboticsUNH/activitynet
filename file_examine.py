import argparse
import os
import numpy as np


def readfile(filename):
    file = open(filename, "rb")
    lines = np.loadtxt(file, dtype=str)
    print(len(lines))
    ret = []
    for line in lines:
        frame = []
        vals = line.split(',')
        for val in vals:
            convert = float(val)
            frame.append(convert)
        ret.append(frame)


def main(args):
    file = os.getcwd() + '/' + args['file']
    if file != None: readfile(file)
    else:
        print("\n*** NO INPUT GIVEN --> TERMINATING PROGRAM ***\n") 
        return None
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Play video and save frame ids for activity recognition')
    parser.add_argument('-f','--file', help='video file', required=False) 
    parser.add_argument('-d','--directory', help='directory name', default=None)
    parser.add_argument('-m', '--mode', help='run mode', required=False)
    args = vars(parser.parse_args())
    main(args)