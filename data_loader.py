import os 
import torch
import pickle  
import numpy as np

from functools import partial
from numpy.random import uniform
from multiprocessing import Process, Manager  

from torch.utils import data
from torch.utils.data.sampler import Sampler

import pdb


class Utterances(data.Dataset):
    """Dataset class for the Utterances dataset."""

    def __init__(self, root_dir, feat_dir, mode):
        """Initialize and preprocess the Utterances dataset."""
        self.root_dir = root_dir
        self.feat_dir = feat_dir
        self.mode = mode
        self.step = 20
        self.split = 0
        
        metaname = os.path.join(self.root_dir, "train.pkl")
        meta = pickle.load(open(metaname, "rb"))
        
        manager = Manager()
        meta = manager.list(meta)
        dataset = manager.list(len(meta)*[None])  # <-- can be shared between processes.
        processes = []
        for i in range(0, len(meta), self.step):
            p = Process(target=self.load_data, 
                        args=(meta[i:i+self.step],dataset,i,mode))  
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
            
        
        # very importtant to do dataset = list(dataset)            
        if mode == 'train':
            self.train_dataset = list(dataset)
            self.num_tokens = len(self.train_dataset)
        elif mode == 'test':
            self.test_dataset = list(dataset)
            self.num_tokens = len(self.test_dataset)
        else:
            raise ValueError
        
        print('Finished loading {} dataset...'.format(mode))
        
        
        
    def load_data(self, submeta, dataset, idx_offset, mode):  
        for k, sbmt in enumerate(submeta):    
            uttrs = len(sbmt)*[None]
            # fill in speaker id and embedding
            uttrs[0] = sbmt[0]
            uttrs[1] = sbmt[1]
            # fill in data
            sp_tmp = np.load(os.path.join(self.root_dir, sbmt[2]))
            f0_tmp = np.load(os.path.join(self.feat_dir, sbmt[2]))
            if self.mode == 'train':
                sp_tmp = sp_tmp[self.split:, :]
                f0_tmp = f0_tmp[self.split:]
            elif self.mode == 'test':
                sp_tmp = sp_tmp[:self.split, :]
                f0_tmp = f0_tmp[:self.split]
            else:
                raise ValueError
            uttrs[2] = ( sp_tmp, f0_tmp )
            dataset[idx_offset+k] = uttrs
            
                   
        
    def __getitem__(self, index):
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        
        list_uttrs = dataset[index]
        spk_id_org = list_uttrs[0]
        emb_org = list_uttrs[1]
        
        melsp, f0_org = list_uttrs[2]
        
        return melsp, emb_org, f0_org
    

    def __len__(self):
        """Return the number of spkrs."""
        return self.num_tokens
    
    

class MyCollator(object):
    def __init__(self, hparams):
        self.min_len_seq = hparams.min_len_seq
        self.max_len_seq = hparams.max_len_seq
        self.max_len_pad = hparams.max_len_pad
        
    def __call__(self, batch):
        # batch[i] is a tuple of __getitem__ outputs
        new_batch = []
        for token in batch:
            aa, b, c = token
            len_crop = np.random.randint(self.min_len_seq, self.max_len_seq+1, size=2) # 1.5s ~ 3s
            left = np.random.randint(0, len(aa)-len_crop[0], size=2)
            pdb.set_trace()
            
            a = aa[left[0]:left[0]+len_crop[0], :]
            c = c[left[0]:left[0]+len_crop[0]]
            
            a = np.clip(a, 0, 1)
            
            a_pad = np.pad(a, ((0,self.max_len_pad-a.shape[0]),(0,0)), 'constant')
            c_pad = np.pad(c[:,np.newaxis], ((0,self.max_len_pad-c.shape[0]),(0,0)), 'constant', constant_values=-1e10)
            
            new_batch.append( (a_pad, b, c_pad, len_crop[0]) ) 
            
        batch = new_batch  
        
        a, b, c, d = zip(*batch)
        melsp = torch.from_numpy(np.stack(a, axis=0))
        spk_emb = torch.from_numpy(np.stack(b, axis=0))
        pitch = torch.from_numpy(np.stack(c, axis=0))
        len_org = torch.from_numpy(np.stack(d, axis=0))
        
        return melsp, spk_emb, pitch, len_org
    


    
class MultiSampler(Sampler):
    """Samples elements more than once in a single pass through the data.
    """
    def __init__(self, num_samples, n_repeats, shuffle=False):
        self.num_samples = num_samples
        self.n_repeats = n_repeats
        self.shuffle = shuffle

    def gen_sample_array(self):
        self.sample_idx_array = torch.arange(self.num_samples, dtype=torch.int64).repeat(self.n_repeats)
        if self.shuffle:
            self.sample_idx_array = self.sample_idx_array[torch.randperm(len(self.sample_idx_array))]
        return self.sample_idx_array

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return len(self.sample_idx_array)        
    
    
    

def get_loader(hparams):
    """Build and return a data loader."""
    
    dataset = Utterances(hparams.root_dir, hparams.feat_dir, hparams.mode)
    
    my_collator = MyCollator(hparams)
    
    sampler = MultiSampler(len(dataset), hparams.samplier, shuffle=hparams.shuffle)
    
    worker_init_fn = lambda x: np.random.seed((torch.initial_seed()) % (2**32))
    
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=hparams.batch_size,
                                  sampler=sampler,
                                  num_workers=hparams.num_workers,
                                  drop_last=True,
                                  pin_memory=True,
                                  worker_init_fn=worker_init_fn,
                                  collate_fn=my_collator)
    return data_loader
