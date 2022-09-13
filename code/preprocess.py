import numpy as np
import scipy.io as io

class Struct(dict):
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value

def load_data(dataset_name):
    filepath = '../data/'+dataset_name+'.mat'
    data_file = io.loadmat(filepath)

    dataset = Struct()
    dataset.probe_feat = np.array(data_file['probe_feat'])
    dataset.gallery_feat = np.array(data_file['gallery_feat'])
    dataset.feedback_score = np.array(1 - data_file['mask']) # we take the groundtruth as feedback scores

    if dataset_name=='oxford5k':
        dataset.eval = data_file['dataset_eval'][0][0]
        dataset.file_list = data_file['file_names']
        dataset.query_idx = np.array(data_file['query_idx'])-1
    else:
        dataset_eval = data_file['dataset_eval'][0][0]
        dataset.query_id = np.array(dataset_eval['queryID'][0][0])
        dataset.test_id = np.array(dataset_eval['testID'][0][0])
        dataset.query_cam = np.array(dataset_eval['queryCAM'][0][0])
        dataset.test_cam = np.array(dataset_eval['testCAM'][0][0])

        dataset.query_id  = dataset.query_id.reshape(len(dataset.query_id))
        dataset.test_id = dataset.test_id.reshape(len(dataset.test_id))
        dataset.query_cam = dataset.query_cam.reshape(len(dataset.query_cam))
        dataset.test_cam = dataset.test_cam.reshape(len(dataset.test_cam))

    dataset.probe_num = dataset.probe_feat.shape[0]
    dataset.gallery_num = dataset.gallery_feat.shape[0]
    dataset.feat_dim = dataset.gallery_feat.shape[1]

    return dataset

def init_params(dataset,t=5,q=5,k=300,alpha=1e-2,method="CAAF"):
    params = Struct()
    params.dataset = dataset # dataset_name
    params.t = t # total query times
    params.q = q # number of feedback samples per round
    params.k = k # size of the new gallery set
    params.alpha = alpha # balancing parameter for manifold ranking
    params.method = method # AL strategy

    return params

def select_top_k(dataset,k,i):
    probe_feat = dataset.probe_feat[i,:] # the i-th probe
    gallery_feat = dataset.gallery_feat

    init_sim = np.dot(gallery_feat,probe_feat.T)
    idx = np.argsort(-init_sim)
    init_rank = np.argsort(idx) # the initial rank of all samples
    idx = idx[0:k] # the original index of the top-k samples

    curr_gallery_feat = gallery_feat[idx,:]
    node_feat = np.vstack((curr_gallery_feat,probe_feat))
    W = np.dot(node_feat,node_feat.T)

    return idx, init_rank, W, node_feat

    
