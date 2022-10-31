import torch
import os
import numpy as np
import time
import sys

torch.set_default_tensor_type(torch.DoubleTensor)
os.environ['CUDA_VISIBLE_DEVICES']='0'

from preprocess import *
from caaf import *
from utils import *


def run(params, dataset):

    dist_mat = np.zeros((dataset.probe_num,dataset.gallery_num,params.t))
    time_cost = np.zeros((dataset.probe_num,params.t))

    # preparation for log
    file_name = '../log/'+params.dataset+'-'+time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    sys.stdout = Logger(filename=file_name+'.txt')
    print(params)

    print("\r", end="")
    print("Progress: {}/{}: ".format(0,dataset.probe_num), "▋" * (0 // 2), end="")
    sys.stdout.flush()

    # test for each probe
    for i in range(0,dataset.probe_num):
        # progress bar
        if (i+1)%10 == 0:
            progress = int((i+1)*100/dataset.probe_num)
            print("\r", end="")
            print("Progress: {}/{}: ".format(i+1,dataset.probe_num), "▋" * (progress // 2), end="")
            sys.stdout.flush()

        # select the top-k samples for retrieval
        idx, init_rank, W, _ = select_top_k(dataset, params.k, i)
        y = np.ones(params.k+1)
        y[:-1] = dataset.feedback_score[idx,i]

        # construct the model
        model = CAAF(W, params.alpha)

        # for each round of feedback
        for j in range(0,params.t):
            time_start=time.time()

            # optimize the model
            temp_f = model.optimize()

            # collect ranking scores
            top_f = temp_f[0:params.k]
            f_idx = np.argsort(-top_f)
            rank_f = np.argsort(f_idx)
            init_rank[idx] = rank_f # only the top-k samples in the inital ranking list are updated
            dist_mat[i,:,j] = init_rank   

            # collect feedback suggestions
            fb_id = model.select(params.q)
            assert len(fb_id) == params.q and set(fb_id).issubset(set(model.unlabeled_gallery_set))
            fb_score = y[fb_id]

            # update the parameters
            model.update(fb_id,fb_score)

            time_end = time.time()
            time_cost[i,j] = time_end-time_start

    # evaluate
    r1 = np.zeros(params.t)
    map = np.zeros(params.t)
    print('')
    for t in range(0,params.t):
        curr_dist_mat = dist_mat[:,:,t]
        if params.dataset == 'oxford5k':
            cmc_score,map_score = evaluate_oxford_5k(curr_dist_mat,dataset.eval,dataset.file_list)
        else:
            cmc_score,map_score = evaluate(-curr_dist_mat, dataset.query_id, dataset.query_cam, dataset.test_id, dataset.test_cam)
        r1[t] = float(cmc_score[0])*100
        map[t] = float(map_score)*100
        print('round: %d\trank@1: %.2f\tmap: %.2f'%(int(t),r1[t],map[t]))
    print('time/round= %.4f'%np.mean(time_cost[:,1:]))
    print('\n-----------------------------------\n')

if __name__ == '__main__':
    dataset = 'oxford5k' #['oxford5k','holidays','cuhk03']
    params = init_params(dataset=dataset,t=5,q=5,k=300,alpha=1e-2,method="CAAF")
    data = load_data(dataset)
    run(params,data)