import numpy as np
import torch


# fmap,gradmap size = [channel,lenth]
def caculate_cam_vlue(fmap,Fc_c_weights,original_seq_lenth = 5000):
    fmap_weights = fmap * Fc_c_weights     
    cam = fmap_weights.sum(axis = 0) 
    cam[cam<0] = 0 # like relu                                   #[lenth,]
    cam = (cam - cam.min())/(1e-7*cam.max()) # maxmin normalize  #[lenth,]
    cam_tensor = torch.tensor(cam)
    cam_tensor = (cam_tensor.unsqueeze(0)).unsqueeze(0)
    upsampler = torch.nn.Upsample(original_seq_lenth,mode='linear',align_corners=False)
    cam_tensor = upsampler(cam_tensor)
    cam = (cam_tensor[0][0]).to('cpu').detach().numpy()
    return cam

# fmap,gradmap size = [channel,lenth]
def caculate_grad_cam_vlue(fmap,gradmap,original_seq_lenth = 5000):
    weights = np.mean(gradmap,axis=1,keepdims=True)              # [channel,1],取每个通道下所有梯度的平均数为该通道的权重
    fmap_weights = fmap * weights                                # [channel,lenth]*[channel,1] = [channel,lenth]
    cam = fmap_weights.sum(axis = 0)                             #[lenth,]
    cam[cam<0] = 0 # like relu                                   #[lenth,]
    cam = (cam - cam.min())/(1e-7*cam.max()) # maxmin normalize  #[lenth,]
    #cam = (cam*gradmap).sum(axis=0) # [channel,lenth]
    # top_idx=cam.argsort()[0:200]#min 500
    # cam[top_idx] = 0
    cam_tensor = torch.tensor(cam)
    cam_tensor = (cam_tensor.unsqueeze(0)).unsqueeze(0)
    upsampler = torch.nn.Upsample(original_seq_lenth,mode='linear',align_corners=False)
    cam_tensor = upsampler(cam_tensor)
    cam = (cam_tensor[0][0]).to('cpu').detach().numpy()
    return cam

    # fmap,gradmap size = [channel,lenth]
def caculate_grad_cam_pp_vlue(fmap,gradmap,original_seq_lenth = 5000):
    #weights = np.mean(gradmap,axis=1,keepdims=True)              # [channel,1],取每个通道下所有梯度的平均数为该通道的权重
    grad_2 = gradmap ** 2
    grad_3 = grad_2 * gradmap
    sum_fmap = fmap.sum(axis=1)
    aij = grad_2/(2*grad_2+sum_fmap[:,None]*grad_3+1e-7)
    aij[aij<0] = 0
    weights =  np.maximum(gradmap, 0) * aij
    weights = (np.sum(weights, axis=1))
    weights = np.expand_dims(weights,axis=1)

    fmap_weights = fmap * weights                                # [channel,lenth]*[channel,1] = [channel,lenth]
    cam = fmap_weights.sum(axis = 0)                             #[lenth,]
    cam[cam<0] = 0 # like relu                                   #[lenth,]
    cam = (cam - cam.min())/(1e-7*cam.max()) # maxmin normalize  #[lenth,]
    #cam = (cam*gradmap).sum(axis=0) # [channel,lenth]
    # top_idx=cam.argsort()[0:200]#min 500
    # cam[top_idx] = 0
    cam_tensor = torch.tensor(cam)
    cam_tensor = (cam_tensor.unsqueeze(0)).unsqueeze(0)
    upsampler = torch.nn.Upsample(original_seq_lenth,mode='linear',align_corners=False)
    cam_tensor = upsampler(cam_tensor)
    cam = (cam_tensor[0][0]).to('cpu').detach().numpy()
    return cam

# fmap,gradmap size = [channel,lenth]
def caculate_layer_cam_vlue(fmap,gradmap,original_seq_lenth = 5000):
    gradmap[gradmap<0] = 0 #[channel,lenth]
    weights = gradmap #[channel,lenth]
    fmap_ = weights*fmap #[channel,lenth]

    cam = np.sum(fmap_,axis=0) #[lenth,]
    cam[cam<0] = 0 # like relu                                   #[lenth,]
    # cam = (cam - cam.min())/(1e-7*cam.max()) # maxmin normalize  # [lenth,]
    cam_tensor = torch.tensor(cam)
    cam_tensor = (cam_tensor.unsqueeze(0)).unsqueeze(0)
    upsampler = torch.nn.Upsample(original_seq_lenth,mode='linear',align_corners=False)
    cam_tensor = upsampler(cam_tensor)
    cam = (cam_tensor[0][0]).to('cpu').detach().numpy()
    return cam


def scaled_layer_cam_tanh(cam,gamma = 2.):
    (cam_min, cam_max) = (cam.min(), cam.max())
    norm_cam = np.tanh(gamma*cam/cam_max)
    return norm_cam

# fmap,gradmap size = [channel,H,W]
def caculate_layer_cam_vlue_2d(fmap,gradmap,original_size = (12,5000),scale_tanh:bool=False):
    gradmap[gradmap<0] = 0 #[channel,H,W] like relu  
    weights = gradmap #[channel,H,W]
    fmap_ = weights*fmap #[channel,H,W]
    cam = np.sum(fmap_,axis=0) #[H,W]
    cam[cam<0] = 0 # like relu #[H,W]
    if(scale_tanh):
        cam = scaled_layer_cam_tanh(cam,gamma=2.)
    # up sampler to original_size(or call target size)
    cam_tensor = torch.tensor(cam)
    cam_tensor = (cam_tensor.unsqueeze(0)).unsqueeze(0) #[H,W]->[1,1,H,W]
    cam_tensor = torch.nn.functional.interpolate(cam_tensor, size=(original_size), mode='bilinear', align_corners=True)
    #normal
    cam = (cam_tensor[0][0]).to('cpu').detach().numpy()
    return cam
    # (cam_min, cam_max) = (cam.min(), cam.max())
    # norm_cam = (cam_tensor - cam_min) / (((cam_max - cam_min) + 1e-08)).data
    # return norm_cam
