import numpy as np 
import os 

def parse_p2s(filepath):
    """ 
    read p2s.txt and represent the numbers as numpy format.
    
    filepath: path to the directory containing p2s.txt
    """

    file_p2s= os.path.join(filepath, "p2s.txt") 
    with open(file_p2s, 'r') as f:
        lines=f.readlines()
        lines=[line.strip().strip(',') for line in lines] #remove \n and last comma

        ses=[]
        for line in lines:
            # print(line)
            se=line.split(',')
            if len(se)<4:
                se=[0]*50
            else:
                se=[int(s) for s in se]
            ses.append(se)

        ses=np.array(ses)
    return ses


def split_eat_noneat(dir_segs, des=30):
    """ 
    dir_segs: dictionary of filepaths to segments
    Returns two dictionaries: 
    eating_file2segs: dictionary of filepaths to segments where the label starts with 'eat'
    not_eating_file2segs: dictionary of filepaths to segments where the label starts with 'not'
    """

    eating_file2segs={}
    not_eating_file2segs={}
    for filepath in dir_segs.keys():
        segs=dir_segs[filepath]
        for s,e,t in segs:
            d=e-s 
            if d<des:
                continue  #skip segments less than 30 frames
            #if t starts with eat
            if t.startswith("eat"):
                if filepath in eating_file2segs:
                    eating_file2segs[filepath].append((s,e,t))
                else:
                    eating_file2segs[filepath]=[(s,e,t)]
            elif t.startswith("not"):
                if filepath in not_eating_file2segs:
                    not_eating_file2segs[filepath].append((s,e,t))
                else:
                    not_eating_file2segs[filepath]=[(s,e,t)]

    return eating_file2segs, not_eating_file2segs

def load_s_e_l_segs(filename):
    """
    start, end, label

    data labelled by watching videos and saved into a txt file.

    Extract segments from a file containing segments.

    return a list of tuples (start, end, label)
    """
    
    segs=[]

    with open(filename) as f:
        lines=f.readlines()
        
    lines=[line.strip() for line in lines]

    for line in lines:
        if 'start' in line:
            continue
        if len(line)<3:
            continue
        line=line.split(",") 
        s,e,t=int(line[0]),int(line[1]), line[2].strip()
        # print(s,e,t)
        segs.append((s,e,t))
    return segs

def segment_lengths(file_dict):
    """ 
    file_dict: dictionary of filepaths and their segments
    return a list of length of all the segments
    """

    durations=[]
    for filepath in file_dict.keys():
        segs=file_dict[filepath]
        for s,e,t in segs:
            durations.append(e-s)

    durations=sorted(durations)
    return durations

def split_into_fixed_length_segments(start, end, length=30, max_overlap=10):
    """ 
    split (start-end) segment into multiple segments of fixed length with a maximum overlap.bit_length


    start: start frame
    end: end frame
    length: length of of the desired segment
    max_overlap: maximum overlap allowed between segments

    generate segments of fixed length with a maximum overlap
    spread out the last segment if to decrease overlap.

    return a list of tuples (start, end) representing the segments
    """

    step = length - max_overlap
    segments = []
    
    current_start = start
    while current_start + length <= end:
        segments.append((current_start, current_start + length))
        current_start += step
    
    # Adjust the start of the last segment if it falls short of the end
    if segments and segments[-1][1] < end:
        last_segment_start = max(start, end - length)
        segments[-1] = (last_segment_start, last_segment_start + length)
    
    return segments
 
def generate_more_segments(file_dict, seq_length=30, max_overlap=10):
    """ 
    Input: take a dictionary of filepaths and their segments
    Output: return a dictionary of filepaths and more segments of fixed length generated from the segments.

    file_dict: {filepath: [(start, end, label), ...]}
    generate segments of fixed length with a maximum overlap
    spread out the last segment if to decrease overlap.
    """
    segs_fixed={}
    for filepath in file_dict.keys():
        segs=file_dict[filepath] 
        for s,e,t in segs: 
            more_segs=split_into_fixed_length_segments(s, e, seq_length, max_overlap)  
            if filepath in segs_fixed:
                segs_fixed[filepath].extend(more_segs)
            else:
                segs_fixed[filepath]=more_segs

    return segs_fixed