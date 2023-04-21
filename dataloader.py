import torch, h5py, os, glob
from torch.utils.data import Dataset
import numpy as np
import sigpy as sp
​
class MCFullFastMRI(Dataset):
    def __init__(self, config):
        self.num_slices       = 5
        self.center_slice     = 2
        self.ACS_size         = 24
        self.ksp_files        = glob.glob(config.data.ksp_path)
        self.ksp_files.remove('/csiNAS/mridata/fastmri_brain/brain_multicoil_train/multicoil_train/file_brain_AXT2_210_2100070.h5')
        if not config.data.train_size:
            config.data.train_size = len(self.ksp_files) * self.num_slices
        
        self.ksp_files        = self.ksp_files[0:int(config.data.train_size / self.num_slices)]
        print("Dataset Size: " + str(config.data.train_size))
        self.maps_dir         = config.data.map_path
​
    def __len__(self):
        return len(self.ksp_files) * self.num_slices
​
    def __getitem__(self, idx):
        # Convert to numerical
        if torch.is_tensor(idx):
            idx = idx.tolist()
​
        # Separate slice and sample
        sample_idx = idx // self.num_slices
        slice_idx  = self.center_slice + np.mod(idx, self.num_slices) - self.num_slices // 2
        
        # Load MRI samples and maps
        with h5py.File(self.ksp_files[sample_idx], 'r') as contents:
            # Get k-space for specific slice
            ksp = np.asarray(contents['kspace'][slice_idx]) # shape = [C,H,W]
            
        map_file = self.maps_dir + os.path.basename(self.ksp_files[sample_idx])
        with h5py.File(map_file, 'r') as contents:
            # Get sensitivity maps for specific slice
            s_maps = np.asarray(contents['s_maps'][slice_idx])#was map_idx
         
        gt_img = self.adjoint_fs(ksp, s_maps) #shape [H,W]
​
        gt_img_cropped = sp.resize(gt_img, [384,384]) # shape [384,384]
        gt_maps_cropped = sp.resize(s_maps, [s_maps.shape[0],384,384]) # shape [C, 384, 384]
​
        # Get normalization constant from undersampled RSS
        gt_ksp_cropped = self.forward_fs(gt_img_cropped[None,...], gt_maps_cropped) # shape [C,384,384]
        # zero out everything but ACS
        gt_ksp_acs_only = sp.resize(sp.resize(gt_ksp_cropped, (s_maps.shape[0], self.ACS_size, self.ACS_size)), gt_ksp_cropped.shape)
        # make RCS img
        ACS_img = sp.rss(sp.ifft(gt_ksp_acs_only, axes =(-2,-1)), axes=(0,))
        norm_const = np.percentile(np.abs(ACS_img), 99)
​
        gt_img_cplx_norm = gt_img_cropped/norm_const
        gt_img_2ch_norm  = torch.view_as_real(torch.tensor(gt_img_cplx_norm)).permute(-1,0,1)
​
        return {'X': gt_img_2ch_norm}
​
    def adjoint_fs(self, ksp, maps):
        coil_imgs = sp.ifft(ksp, axes = (-2,-1))
        coil_imgs_with_maps = coil_imgs*np.conj(maps)
        img_out = np.sum(coil_imgs_with_maps, axis = -3)
        return img_out
​
    def forward_fs(self, img, maps):
        coil_imgs = img*maps
        coil_ksp = sp.fft(coil_imgs, axes = (-2,-1))
        return coil_ksp