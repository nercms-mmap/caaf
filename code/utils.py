import numpy as np
import sys

class Logger(object):
    def __init__(self, filename, stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def evaluate(sim_score,qlabel,qcam,glabel,gcam):
    ap = 0
    cmc = np.zeros(len(glabel))

    for i in range(len(qlabel)):
        ql = qlabel[i]
        qc = qcam[i]
        score = sim_score[i,:]
        # predict index
        index = np.argsort(score)  #from small to large
        index = index[::-1]
        #index = index[0:2000]
        # good index
        query_index = np.argwhere(glabel==ql)
        camera_index = np.argwhere(gcam==qc)

        good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
        junk_index1 = np.argwhere(glabel==-1)
        junk_index2 = np.intersect1d(query_index, camera_index)
        junk_index = np.append(junk_index2, junk_index1) #.flatten())
        
        ap_tmp,cmc_tmp = compute_AP(index, good_index, junk_index)
        if cmc_tmp[0]==-1:
            continue
        cmc = cmc + cmc_tmp
        ap += ap_tmp

    cmc = cmc/len(qlabel)
    map = ap/len(qlabel)

    return cmc,map


def compute_AP(index, good_index, junk_index):
    ap = 0
    cmc = np.zeros(len(index))
    if good_index.size==0:   # if empty
        cmc[0] = -1
        return ap,cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i]!=0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision=1.0
        ap = ap + d_recall*(old_precision + precision)/2

    return ap, cmc

def evaluate_oxford_5k(dist,eval_para,file_list):
    probe_num = dist.shape[0]
    ap = 0
    cmc = np.zeros(dist.shape[1])

    for k in range(0,probe_num):
        pos_set = eval_para[k][0]['pos_set'][0][0]
        junk_set = eval_para[k][0]['junk_set'][0][0]

        dist_score = dist[k,:]
        idx = np.argsort(dist_score)
        rank_list = file_list[idx]
        
        ap_tmp, cmc_tmp = compute_AP_oxford5k(pos_set, junk_set, rank_list)
    
        if cmc_tmp[0]==-1:
            continue
        cmc = cmc + cmc_tmp
        ap += ap_tmp

    cmc = cmc/probe_num
    map = ap/probe_num

    return cmc,map

def compute_AP_oxford5k(pos_set,junk_set,rank_list):
    cmc = np.zeros(len(rank_list))
    ngood = len(pos_set)

    old_recall = 0 
    old_precision = 1.0
    ap = 0
    intersect_size = 0
    j = 0
    good_now = 0
    njunk = 0
    for n in range(0,len(rank_list)):
        curr_img = rank_list[n].strip()
        #curr_img = char(curr_img)
        flag = 0
        if curr_img in pos_set:
            cmc[n-njunk:-1] = 1
            flag = 1 # good image 
            good_now = good_now+1
        if curr_img in junk_set:
            njunk = njunk + 1
            continue # junk image 
        
        if flag == 1: # good
            intersect_size = intersect_size + 1

        recall = intersect_size/ngood
        precision = intersect_size/(j + 1)
        ap = ap + (recall - old_recall)*((old_precision+precision)/2)
        old_recall = recall
        old_precision = precision
        j = j+1
        
        if good_now == ngood:
            return ap,cmc
        
