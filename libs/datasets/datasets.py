import os
import torch
from .data_utils import trivial_batch_collator, worker_init_reset_seed

datasets = {}
def register_dataset(name):
   def decorator(cls):
       datasets[name] = cls
       return cls
   return decorator

def make_dataset(name, is_training, split, backbone_type, round, **kwargs):
   """
       A simple dataset builder
   """
   dataset = datasets[name](is_training, split, backbone_type, round, **kwargs)
   return dataset

def make_data_loader(dataset, is_training, generator, batch_size, num_workers):
    """
        A simple dataloder builder
    """
    # Cap workers for video loading stability
    safe_num_workers = min(num_workers, 2)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=safe_num_workers,
        collate_fn=trivial_batch_collator,
        worker_init_fn=(worker_init_reset_seed if is_training else None),
        shuffle=is_training,
        drop_last=is_training,
        generator=generator,
        persistent_workers=False,  # Restart workers to prevent memory leaks
        pin_memory=True,           # Faster CPU-GPU transfer
        prefetch_factor=2 if safe_num_workers > 0 else None,
        timeout=300,  # 5 minutes timeout per batch
    )
    return loader

def make_data_loader_distributed(dataset, sampler, is_training, generator, batch_size, num_workers):
    """
        A simple dataloder builder
    """
    # With 4 GPUs x N workders = 4N parallel video decoders competing for resources.
    safe_num_workers = min(num_workers, 2)  # Cap at 2 workers for video loading stability
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=safe_num_workers,
        collate_fn=trivial_batch_collator,
        worker_init_fn=(worker_init_reset_seed if is_training else None),
        # shuffle=is_training,
        drop_last=is_training,
        generator=generator,
        persistent_workers=False,  # Restart workers to prevent memory leaks
        pin_memory=True,           # Faster CPU-GPU transfer
        prefetch_factor=2 if safe_num_workers > 0 else None,
        sampler=sampler,
        timeout=300,  # 5 minutes timeout per batch (some videos are slow)
    )
    return loader
