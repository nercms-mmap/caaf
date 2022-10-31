import numpy as np
import torch

torch.set_default_tensor_type(torch.DoubleTensor)

class CAAF:
    def __init__(self, W, alpha=1e-2, device=None):
        self.alpha = alpha # balancing parameter in manifold ranking loss
        
        if device is None:
            device = torch.device(0) if torch.cuda.is_available() else torch.device('cpu') 
            
        self.W = torch.as_tensor(W, device=device)
        f = self.W[:,-2:-1]
        self.f = (f - torch.min(f))/(torch.max(f)-torch.min(f))

        # take the probe as a specially labeled sample
        m = W.shape[0]
        self.labeled_gallery_set = np.array([m-1]) # probe index: m
        self.unlabeled_gallery_set = np.arange(m-1) # gallery index: 0 ~ m-1
        
        self.v = torch.zeros_like(f)
        self.v[-1] = 1
        
        self.y = torch.zeros_like(f)
        self.y[-1] = 1

        self.D_norm = 1/torch.sqrt(torch.sum(self.W,axis=1, keepdim=True))

    def solve_f(self):
        W = self.W
        n = self.W.shape[0]
        y = self.y
        v = self.v
        labeled_gallery_set = self.labeled_gallery_set

        # enforce the ranking scores of labled samples approach their feedback scores
        alpha = torch.empty_like(y)
        alpha.fill_(self.alpha)
        alpha[labeled_gallery_set] = 1e6
        
        V_tilde = v + v.t()
        W_tilde = V_tilde*W
        D_tilde = torch.sum(W_tilde,axis=1, keepdim=True)
        D_hat = self.D_norm*self.D_norm*D_tilde
        
        W_hat = self.D_norm*self.D_norm.t()*W_tilde
        Q_hat = torch.diag(alpha.squeeze()*torch.sum(V_tilde,axis=1))
        P_hat = torch.diag(D_hat.squeeze()) - W_hat

        A = P_hat + Q_hat + torch.eye(n, device=P_hat.device)*1e-6
        P = A + A.t()
        q = 2*torch.mm(Q_hat,y)

        f = torch.mm(torch.inverse(P),q)
        self.f = (f - torch.min(f))/(torch.max(f)-torch.min(f))
        self.f[labeled_gallery_set] = self.y[labeled_gallery_set]
    
    def solve_v(self):
        W = self.W
        f = self.f
        y = self.y

        # calculate the confidence sore
        f_norm = self.D_norm*f
        ff = f_norm - f_norm.t()
        smooth_loss = W*ff*ff
        fy = self.alpha*(f-y)*(f-y)
        fitting_loss = fy + fy.t()
        loss_mat = smooth_loss + fitting_loss
        self.v = -torch.sum(loss_mat,axis=1)

    def optimize(self):

        # ranking step
        self.solve_f()
        # suggestion step
        self.solve_v()

        return self.f.cpu().numpy().squeeze()
  
    def select(self,num):

        v = self.v
        unlabeled_id = self.unlabeled_gallery_set

        # select samples with the lowest confidence
        ix = torch.argsort(v[unlabeled_id]) 
        fb_id = unlabeled_id[ix[0:num].tolist()]

        return fb_id

    # update model parameters
    def update(self, fb_id, fb_score):

        self.labeled_gallery_set = np.concatenate((self.labeled_gallery_set,fb_id))
        self.unlabeled_gallery_set = np.setdiff1d(self.unlabeled_gallery_set, self.labeled_gallery_set,True)

        # labeled samples are endowed with high confidence
        self.v = torch.zeros_like(self.y)
        self.v[self.labeled_gallery_set] = 1
        self.y[fb_id] = torch.as_tensor(fb_score, device=self.y.device).unsqueeze(1)