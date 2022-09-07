import numpy as np
import torch

class CAAF:
    def __init__(self,W,alpha=1e-2):
        torch.set_default_tensor_type(torch.DoubleTensor)
        n = W.shape[0] 
        self.alpha = alpha # the balancing parameter in manifold ranking loss
        self.W = torch.from_numpy(W).cuda() # the affinity matrix
        
        # initalize ranking scores with W
        f = self.W[:,-1]
        f = f.reshape((n,1))
        self.f = (f - torch.min(f))/(torch.max(f)-torch.min(f))

        # take the probe as a specially labeled sample
        self.labeled_gallery_set = np.array([n-1]) # probe index: n
        self.unlabeled_gallery_set = np.arange(n-1) # gallery index: 0 ~ n-1
        self.v = torch.zeros((n,1)).cuda() # confidence scores
        self.v[-1] = 1
        self.y = torch.zeros((n,1)).cuda() # reference scores
        self.y[-1] = 1

        # to accelerate inverse calculation
        temp = 1/torch.sqrt(torch.sum(self.W,axis=1))
        self.D_norm = torch.diag(temp)

    def solve_f(self):
        W = self.W
        n = len(self.y)
        y = self.y
        v = self.v
        labeled_gallery_set = self.labeled_gallery_set

        # enforce the ranking scores of labled samples approach their feedback scores
        alpha = self.alpha*np.ones(n)
        alpha[labeled_gallery_set] = 1e6
        alpha = torch.from_numpy(alpha).cuda()

        V_tilde = torch.tile(v,(1,n)) + torch.tile(v.t(),(n,1))
        W_tilde = V_tilde*W
        D_tilde = torch.diag(torch.sum(W_tilde,axis=1))

        D_hat = torch.mm(torch.mm(self.D_norm,D_tilde),self.D_norm)
        W_hat = torch.mm(torch.mm(self.D_norm,W_tilde),self.D_norm)
        Q_hat = torch.diag(alpha*torch.sum(V_tilde,axis=1))
        P_hat = D_hat - W_hat

        A = P_hat + Q_hat + torch.eye(n).cuda()*1e-6
        P = A + A.t()
        q = 2*torch.mm(Q_hat,y)

        f = torch.mm(torch.inverse(P),q)
        self.f = (f - torch.min(f))/(torch.max(f)-torch.min(f))
        self.f[labeled_gallery_set] = self.y[labeled_gallery_set]
    
    def solve_v(self):
        W = self.W
        f = self.f
        y = self.y
        n = len(y)

        # calculate the confidence sore
        f_norm = torch.mm(self.D_norm,f)
        ff = torch.tile(f_norm.reshape((n,1)),(1,n)) - torch.tile(f_norm.t(),(n,1))
        smooth_loss = W*ff*ff
        fy = self.alpha*(f-y)*(f-y)
        fy = fy.reshape((n,1))
        fitting_loss = torch.tile(fy,(1,n)) + torch.tile(fy.t(),(n,1))
        loss_mat = smooth_loss + fitting_loss
        self.v = -torch.sum(loss_mat,axis=1)

    def optimize(self):

        # ranking step
        self.solve_f()
        f = self.f.detach().cpu().numpy()

        # suggestion step
        self.solve_v()

        return f.reshape((len(f),))
  
    def select(self,num):

        v = self.v.detach().cpu().numpy()
        unlabeled_id = self.unlabeled_gallery_set

        # select samples with the lowest confidence
        ix = np.argsort(v[unlabeled_id]) 
        fb_id = unlabeled_id[ix[0:num]]

        return fb_id

    def update(self, fb_id, fb_score):
        self.labeled_gallery_set = np.concatenate((self.labeled_gallery_set,fb_id))
        self.unlabeled_gallery_set = np.setdiff1d(self.unlabeled_gallery_set, self.labeled_gallery_set,True)

        # labeled samples are endowed with high confidence
        self.v = torch.zeros(self.y.shape).cuda()
        self.v[self.labeled_gallery_set] = 1

        n = len(fb_id)
        self.y[fb_id] = torch.Tensor(fb_score).reshape((n,1)).cuda()