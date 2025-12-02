import numpy as np
import torch
import torch.nn as nn


class Buffer(nn.Module):
    def __init__(self, args, input_size=None):
        super().__init__()
        self.args = args
        self.k = 0.03

        self.place_left = True

        if input_size is None:
            self.input_size = args.input_size
        else:
            self.input_size = input_size

        buffer_size = args.buffer_size
        print('buffer has %d slots' % buffer_size)

        bx = torch.FloatTensor(buffer_size, *self.input_size).fill_(0)
        print("bx", bx.shape)
        by = torch.LongTensor(buffer_size).fill_(0)
        bt = torch.LongTensor(buffer_size).fill_(0)
        bu = torch.FloatTensor(buffer_size).fill_(0)

        # logits = torch.FloatTensor(buffer_size, args.n_classes).fill_(0)
        # feature = torch.FloatTensor(buffer_size, 512).fill_(0)

        bx = bx.cuda()
        by = by.cuda()
        bt = bt.cuda()
        bu = bu.cuda()
        # logits = logits.cuda()
        # feature = feature.cuda()
        self.save_logits = None

        self.current_index = 0
        self.n_seen_so_far = 0
        self.is_full = 0

        # registering as buffer allows us to save the object using `torch.save`
        self.register_buffer('bx', bx)
        self.register_buffer('by', by)
        self.register_buffer('bt', bt)
        self.register_buffer('bu', bu)
        # self.register_buffer('logits', logits)
        # self.register_buffer('feature', feature)
        self.to_one_hot = lambda x: x.new(x.size(0), args.n_classes).fill_(0).scatter_(1, x.unsqueeze(1), 1)
        self.arange_like = lambda x: torch.arange(x.size(0)).to(x.device)
        self.shuffle = lambda x: x[torch.randperm(x.size(0))]

        # === Prototype maintenance state ===
        # number of classes
        self.num_classes = args.n_classes
        # per-class sample counts in buffer (updated on add/replace)
        self.register_buffer('class_counts', torch.zeros(self.num_classes, dtype=torch.long, device=self.by.device))
        # per-model storage: model_key -> prototypes [C, D]
        self.proto = {}
        # per-model last features per-sample: model_key -> [buffer_size, D]
        self.last_feat = {}
        # per-model feature dims
        self.feat_dim = {}

    @property
    def x(self):
        return self.bx[:self.current_index]

    @property
    def y(self):
        return self.to_one_hot(self.by[:self.current_index])
    
    @property
    def y_int(self):
        return self.by[:self.current_index]

    @property
    def t(self):
        return self.bt[:self.current_index]

    @property
    def u(self):
        return self.bu[:self.current_index]

    @property
    def valid(self):
        return self.is_valid[:self.current_index]
    
    @property
    def n_bits(self):
        total = 0
        if self.args.buffer_size == 0:
            return total
        for name, buf in self.named_buffers():
            if buf.dtype == torch.float32:
                bits_per_item = 8 if name == 'bx' else 32
            elif buf.dtype == torch.int64:
                bits_per_item = buf.max().float().log2().clamp_(min=1).int().item()

            total += bits_per_item * buf.numel()
        return total
    
    @torch.no_grad()
    def update_uncertainty(self, indices, values):
        if indices is None:
            return
        if isinstance(indices, list):
            if len(indices) == 0:
                return
            indices = torch.tensor(indices, device=self.bu.device, dtype=torch.long)
        elif isinstance(indices, torch.Tensor):
            if indices.numel() == 0:
                return
            indices = indices.to(self.bu.device, dtype=torch.long)
        else:
            indices = torch.tensor([indices], device=self.bu.device, dtype=torch.long)

        values = values.to(self.bu.device).float()
        if values.dim() == 0:
            values = values.unsqueeze(0)
        if values.numel() != indices.numel():
            raise ValueError("Uncertainty length mismatch with indices")
        self.bu.index_copy_(0, indices, values)

    def uncertainty_stats(self):
        if self.current_index == 0:
            return {'mean': 0.0, 'max': 0.0, 'min': 0.0}
        current_u = self.bu[:self.current_index]
        return {
            'mean': float(current_u.mean().item()),
            'max': float(current_u.max().item()),
            'min': float(current_u.min().item()),
        }

    def __len__(self):
        return self.current_index

    def display(self, gen=None, epoch=-1):
        from PIL import Image
        from torchvision.utils import save_image

        if 'cifar' in self.args.dataset:
            shp = (-1, 3, 32, 32)
        elif 'tinyimagenet' in self.args.dataset:
            shp = (-1, 3, 64, 64)
        else:
            shp = (-1, 1, 28, 28)

        if gen is not None:
            x = gen.decode(self.x)
        else:
            x = self.x

        save_image((x.reshape(shp) * 0.5 + 0.5), 'samples/buffer_%d.png' % epoch, nrow=int(self.current_index ** 0.5))
        # Image.open('buffer_%d.png' % epoch).show()
        print(self.y.sum(dim=0))

    def add_reservoir(self, x, y, logits, t, uncertainty=None, u_buffer=None):
        n_elem = x.size(0)
        if n_elem == 0:
            return

        device = self.bx.device
        x = x.to(device)
        y = y.to(device)

        if uncertainty is None:
            uncertainty = torch.zeros(n_elem, device=device)
        else:
            uncertainty = uncertainty.to(device).float()
            if uncertainty.dim() == 0:
                uncertainty = uncertainty.unsqueeze(0)
            if uncertainty.numel() != n_elem:
                raise ValueError("Uncertainty tensor length must match number of samples")

        save_logits = logits is not None
        self.save_logits = logits is not None

        place_left = max(0, self.bx.size(0) - self.current_index)
        result_info = {
            'added': None,
            'added_src_pos': None,
            'replaced': None,
            'replaced_src_pos': None,
            'replaced_old_labels': None,
        }

        if place_left:
            offset = min(place_left, n_elem)
            upper = self.current_index + offset
            self.bx[self.current_index: upper].data.copy_(x[:offset])
            self.by[self.current_index: upper].data.copy_(y[:offset])
            self.bt[self.current_index: upper].fill_(t)
            self.bu[self.current_index: upper].data.copy_(uncertainty[:offset].to(self.bu.device))

            with torch.no_grad():
                add_counts = torch.bincount(y[:offset].view(-1).long().cpu(), minlength=self.num_classes)
                add_counts = add_counts.to(self.class_counts.device)
                self.class_counts[:add_counts.numel()] += add_counts

            added_indices_fill = torch.arange(self.current_index, upper, device=device, dtype=torch.long)
            result_info['added'] = added_indices_fill
            result_info['added_src_pos'] = torch.arange(0, offset, device=device, dtype=torch.long)

            self.current_index += offset
            self.n_seen_so_far += offset

            if offset == n_elem:
                return result_info

        self.place_left = False

        x = x[place_left:]
        y = y[place_left:]
        uncertainty = uncertainty[place_left:]

        indices = torch.empty(x.size(0), device=device).uniform_(0, self.n_seen_so_far).long()
        valid_mask = (indices < self.bx.size(0))

        idx_new_data = valid_mask.nonzero().squeeze(-1)
        idx_buffer = indices[idx_new_data]

        self.n_seen_so_far += x.size(0)

        if idx_buffer.numel() == 0:
            return result_info

        assert idx_buffer.max() < self.bx.size(0)
        assert idx_buffer.max() < self.by.size(0)
        assert idx_buffer.max() < self.bt.size(0)

        assert idx_new_data.max() < x.size(0)
        assert idx_new_data.max() < y.size(0)

        with torch.no_grad():
            old_labels = self.by[idx_buffer].clone()
            old_tasks = self.bt[idx_buffer].clone()
            old_uncertainty = self.bu[idx_buffer].clone()
            old_samples = self.bx[idx_buffer].clone()

            for model_key, proto in self.proto.items():
                if model_key not in self.last_feat:
                    continue
                lf = self.last_feat[model_key]
                for pos in range(idx_buffer.numel()):
                    buf_idx = idx_buffer[pos]
                    c_old = int(old_labels[pos].item())
                    n_c_with = int(self.class_counts[c_old].item())
                    if n_c_with > 0 and lf.size(0) > buf_idx:
                        proto[c_old] = proto[c_old] - lf[buf_idx] / float(n_c_with)
                        lf[buf_idx].zero_()

            dec_counts = torch.bincount(old_labels.view(-1).long().cpu(), minlength=self.num_classes)
            dec_counts = dec_counts.to(self.class_counts.device)
            self.class_counts[:dec_counts.numel()] -= dec_counts

        self.bx[idx_buffer] = x[idx_new_data].to(device)
        self.by[idx_buffer] = y[idx_new_data].to(device)
        self.bt[idx_buffer].fill_(t)
        self.bu[idx_buffer] = uncertainty[idx_new_data].to(self.bu.device)

        with torch.no_grad():
            inc_counts = torch.bincount(y[idx_new_data].view(-1).long().cpu(), minlength=self.num_classes)
            inc_counts = inc_counts.to(self.class_counts.device)
            self.class_counts[:inc_counts.numel()] += inc_counts

        if u_buffer is not None:
            mask_current_task = (old_tasks == t)
            if mask_current_task.any():
                u_buffer.add_batch(
                    old_samples[mask_current_task].to(u_buffer.device),
                    old_labels[mask_current_task].to(u_buffer.device),
                    old_uncertainty[mask_current_task].to(u_buffer.device),
                )

        result_info['replaced'] = idx_buffer
        result_info['replaced_src_pos'] = idx_new_data
        result_info['replaced_old_labels'] = old_labels

        if result_info['added'] is None:
            result_info['added'] = idx_buffer
            result_info['added_src_pos'] = idx_new_data
        else:
            result_info['added'] = torch.cat([result_info['added'], idx_buffer], dim=0)
            result_info['added_src_pos'] = torch.cat([result_info['added_src_pos'], idx_new_data], dim=0)

        return result_info

    def add_priority(self, x, y, t):
        """
        Directly insert samples: fill remaining slots; if full, overwrite random indices.
        This bypasses reservoir sampling to guarantee inclusion.
        """
        n_elem = x.size(0)
        if n_elem == 0:
            return

        place_left = max(0, self.bx.size(0) - self.current_index)
        if place_left:
            offset = min(place_left, n_elem)
            self.bx[self.current_index: self.current_index + offset].data.copy_(x[:offset])
            self.by[self.current_index: self.current_index + offset].data.copy_(y[:offset])
            self.bt[self.current_index: self.current_index + offset].fill_(t)
            self.current_index += offset
            self.n_seen_so_far += offset
            if offset == n_elem:
                return
            # still have remaining to insert
            x = x[offset:]
            y = y[offset:]
            n_elem = x.size(0)

        # buffer is full: overwrite random indices
        idx_buffer = torch.randint(low=0, high=self.bx.size(0), size=(n_elem,), device=x.device)
        self.bx[idx_buffer] = x
        self.by[idx_buffer] = y
        self.bt[idx_buffer] = t

    def measure_valid(self, generator, classifier):
        with torch.no_grad():
            # fetch valid examples
            valid_indices = self.valid.nonzero()
            valid_x, valid_y = self.bx[valid_indices], self.by[valid_indices]
            one_hot_y = self.to_one_hot(valid_y.flatten())

            hid_x = generator.idx_2_hid(valid_x)
            x_hat = generator.decode(hid_x)

            logits = classifier(x_hat)
            _, pred = logits.max(dim=1)
            one_hot_pred = self.to_one_hot(pred)
            correct = one_hot_pred * one_hot_y

            per_class_correct = correct.sum(dim=0)
            per_class_deno = one_hot_y.sum(dim=0)
            per_class_acc = per_class_correct.float() / per_class_deno.float()
            self.class_weight = 1. - per_class_acc
            self.valid_acc = per_class_acc
            self.valid_deno = per_class_deno

    def shuffle_(self):
        indices = torch.randperm(self.current_index).to(self.args.device)
        self.bx = self.bx[indices]
        self.by = self.by[indices]
        self.bt = self.bt[indices]

    def delete_up_to(self, remove_after_this_idx):
        self.bx = self.bx[:remove_after_this_idx]
        self.by = self.by[:remove_after_this_idx]
        self.bt = self.bt[:remove_after_this_idx]
        self.bu = self.bu[:remove_after_this_idx]
        self.current_index = min(self.current_index, remove_after_this_idx)

    def sample(self, amt, exclude_task=None, ret_ind=False):
        if amt <= 0 or self.current_index == 0:
            empty_x = self.bx.new_empty((0,) + self.bx.shape[1:])
            empty_y = self.by.new_empty((0,), dtype=self.by.dtype)
            empty_t = self.bt.new_empty((0,), dtype=self.bt.dtype)
            if ret_ind:
                empty_idx = torch.empty(0, device=self.bx.device, dtype=torch.long)
                return empty_x, empty_y, empty_t, empty_idx
            return empty_x, empty_y, empty_t

        device = self.bx.device
        valid_indices = torch.arange(self.current_index, device=device, dtype=torch.long)

        if exclude_task is not None:
            mask = (self.bt[:self.current_index] != exclude_task)
            valid_indices = valid_indices[mask]

        if valid_indices.numel() == 0:
            empty_x = self.bx.new_empty((0,) + self.bx.shape[1:])
            empty_y = self.by.new_empty((0,), dtype=self.by.dtype)
            empty_t = self.bt.new_empty((0,), dtype=self.bt.dtype)
            if ret_ind:
                empty_idx = torch.empty(0, device=device, dtype=torch.long)
                return empty_x, empty_y, empty_t, empty_idx
            return empty_x, empty_y, empty_t

        sample_count = min(int(amt), valid_indices.numel())
        if sample_count == 0:
            empty_x = self.bx.new_empty((0,) + self.bx.shape[1:])
            empty_y = self.by.new_empty((0,), dtype=self.by.dtype)
            empty_t = self.bt.new_empty((0,), dtype=self.bt.dtype)
            if ret_ind:
                empty_idx = torch.empty(0, device=device, dtype=torch.long)
                return empty_x, empty_y, empty_t, empty_idx
            return empty_x, empty_y, empty_t

        if sample_count < valid_indices.numel():
            perm = torch.randperm(valid_indices.numel(), device=device)[:sample_count]
            chosen_indices = valid_indices[perm]
        else:
            chosen_indices = valid_indices

        bx_sel = self.bx[chosen_indices]
        by_sel = self.by[chosen_indices]
        bt_sel = self.bt[chosen_indices]

        if ret_ind:
            return bx_sel, by_sel, bt_sel, chosen_indices
        return bx_sel, by_sel, bt_sel

    def split(self, amt):
        indices = torch.randperm(self.current_index).to(self.args.device)
        return indices[:amt], indices[amt:]

    def onlysample(self, amt, task=None, ret_ind=False):

        if amt <= 0 or self.current_index == 0:
            empty_x = self.bx.new_empty((0,) + self.bx.shape[1:])
            empty_y = self.by.new_empty((0,), dtype=self.by.dtype)
            empty_t = self.bt.new_empty((0,), dtype=self.bt.dtype)
            if ret_ind:
                empty_idx = torch.empty(0, device=self.bx.device, dtype=torch.long)
                return empty_x, empty_y, empty_t, empty_idx
            return empty_x, empty_y, empty_t

        device = self.bx.device
        valid_indices = torch.arange(self.current_index, device=device, dtype=torch.long)

        if task is not None:
            mask = (self.bt[:self.current_index] == task)
            valid_indices = valid_indices[mask]

        if valid_indices.numel() == 0:
            empty_x = self.bx.new_empty((0,) + self.bx.shape[1:])
            empty_y = self.by.new_empty((0,), dtype=self.by.dtype)
            empty_t = self.bt.new_empty((0,), dtype=self.bt.dtype)
            if ret_ind:
                empty_idx = torch.empty(0, device=device, dtype=torch.long)
                return empty_x, empty_y, empty_t, empty_idx
            return empty_x, empty_y, empty_t

        sample_count = min(int(amt), valid_indices.numel())
        if sample_count == 0:
            empty_x = self.bx.new_empty((0,) + self.bx.shape[1:])
            empty_y = self.by.new_empty((0,), dtype=self.by.dtype)
            empty_t = self.bt.new_empty((0,), dtype=self.bt.dtype)
            if ret_ind:
                empty_idx = torch.empty(0, device=device, dtype=torch.long)
                return empty_x, empty_y, empty_t, empty_idx
            return empty_x, empty_y, empty_t

        if sample_count < valid_indices.numel():
            perm = torch.randperm(valid_indices.numel(), device=device)[:sample_count]
            chosen_indices = valid_indices[perm]
        else:
            chosen_indices = valid_indices

        bx_sel = self.bx[chosen_indices]
        by_sel = self.by[chosen_indices]
        bt_sel = self.bt[chosen_indices]

        if ret_ind:
            return bx_sel, by_sel, bt_sel, chosen_indices
        return bx_sel, by_sel, bt_sel

    # === Prototype maintenance APIs ===
    @torch.no_grad()
    def ensure_model_storage(self, model_key, feat_dim):
        if model_key in self.proto:
            return
        device = self.bx.device
        self.proto[model_key] = torch.zeros(self.num_classes, feat_dim, device=device)
        self.last_feat[model_key] = torch.zeros(self.bx.size(0), feat_dim, device=device)
        self.feat_dim[model_key] = feat_dim

    @torch.no_grad()
    def update_on_add(self, indices, labels, features_by_model):
        # indices: global buffer indices (LongTensor)
        # labels: LongTensor aligned with indices
        if indices is None or len(indices) == 0:
            return
        for model_key, feats in features_by_model.items():
            if feats is None:
                continue
            self.ensure_model_storage(model_key, feats.size(1))
            for pos in range(indices.numel()):
                buf_idx = int(indices[pos].item())
                c = int(labels[pos].item())
                n_c = int(self.class_counts[c].item())
                if n_c <= 0:
                    continue
                self.proto[model_key][c] = self.proto[model_key][c] + feats[pos] / float(n_c)
                self.last_feat[model_key][buf_idx] = feats[pos]

    @torch.no_grad()
    def update_on_refresh(self, indices, labels, features_by_model):
        # indices: global buffer indices (LongTensor)
        # labels: LongTensor aligned with indices
        if indices is None or len(indices) == 0:
            return
        for model_key, feats in features_by_model.items():
            if feats is None or model_key not in self.proto:
                # if storage not initialized for this model, initialize now
                if feats is not None:
                    self.ensure_model_storage(model_key, feats.size(1))
                else:
                    continue
            lf = self.last_feat[model_key]
            proto = self.proto[model_key]
            for pos in range(indices.numel()):
                buf_idx = int(indices[pos].item())
                c = int(labels[pos].item())
                n_c = int(self.class_counts[c].item())
                if n_c <= 0:
                    continue
                delta = feats[pos] - lf[buf_idx]
                proto[c] = proto[c] + delta / float(n_c)
                lf[buf_idx] = feats[pos]
        
    def print_per_task_num(self):
        _, counts = torch.unique(self.bt, return_counts=True)
        print(f"Number of buffed imgs: {counts.tolist()}")