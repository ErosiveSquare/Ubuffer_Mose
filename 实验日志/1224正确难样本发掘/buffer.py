import torch
import torch.nn as nn
import numpy as np


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

        # statistics cache
        self.latest_u_mean = 0.0

        # === Prototype maintenance state ===
        # number of classes
        self.num_classes = args.n_classes
        # per-class sample counts in buffer (updated on add/replace)
        self.register_buffer('class_counts', torch.zeros(self.num_classes, dtype=torch.long, device=self.by.device))
        
        # 类别 -> 缓冲区索引映射（CPU维护，便于快速采样）
        self.class_to_indices = {i: [] for i in range(self.num_classes)}
        # 每类前 m 个索引缓存
        self.class_first_m = {i: [] for i in range(self.num_classes)}
        self.class_first_m_cap = 10

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
    
    # === 索引映射维护 ===
    def _refresh_first_m(self, cls_id):
        lst = self.class_to_indices.get(int(cls_id), [])
        if lst is None:
            lst = []
        self.class_first_m[int(cls_id)] = sorted(lst)[: self.class_first_m_cap]

    def _update_class_index_maps(
        self,
        add_idx_cpu=None,
        add_labels_cpu=None,
        remove_idx_cpu=None,
        remove_labels_cpu=None,
    ):
        affected = set()
        # 移除旧索引
        if remove_idx_cpu is not None and remove_labels_cpu is not None:
            for idx, label in zip(remove_idx_cpu.tolist(), remove_labels_cpu.tolist()):
                lst = self.class_to_indices.get(int(label), [])
                if idx in lst:
                    lst.remove(idx)
                affected.add(int(label))
        # 添加新索引（若已存在则跳过，避免重复）
        if add_idx_cpu is not None and add_labels_cpu is not None:
            for idx, label in zip(add_idx_cpu.tolist(), add_labels_cpu.tolist()):
                lst = self.class_to_indices.get(int(label), [])
                if idx not in lst:
                    lst.append(int(idx))
                    self.class_to_indices[int(label)] = lst
                    affected.add(int(label))
        # 刷新前 m
        for cls_id in affected:
            self._refresh_first_m(cls_id)

    def _rebuild_class_index_maps(self):
        self.class_to_indices = {i: [] for i in range(self.num_classes)}
        self.class_first_m = {i: [] for i in range(self.num_classes)}
        labels = self.by[: self.current_index].detach().cpu().tolist()
        for idx, label in enumerate(labels):
            self.class_to_indices[int(label)].append(int(idx))
        for cls_id in range(self.num_classes):
            self._refresh_first_m(cls_id)

    def sample_class_data(self, class_id: int, n_samples: int, target_device=None, exclude_indices=None):
        if n_samples <= 0:
            return None, None, None
        exclude_set = set(exclude_indices) if exclude_indices is not None else set()
        idx_list = [idx for idx in self.class_to_indices.get(int(class_id), []) if idx not in exclude_set and idx < self.current_index]
        if len(idx_list) == 0:
            return None, None, None
        if len(idx_list) > n_samples:
            idx_list = np.random.choice(idx_list, n_samples, replace=False).tolist()
        idx_tensor = torch.tensor(idx_list, device=self.bx.device, dtype=torch.long)
        x_sel = self.bx[idx_tensor]
        y_sel = self.by[idx_tensor]
        if target_device is not None:
            x_sel = x_sel.to(target_device)
            y_sel = y_sel.to(target_device)
        return x_sel, y_sel, idx_tensor.cpu()

    def sample_from_classes(self, class_list: list, n_samples: int, target_device=None, exclude_indices=None):
        if n_samples <= 0 or len(class_list) == 0:
            return None, None, None
        exclude_set = set(exclude_indices) if exclude_indices is not None else set()
        candidate_indices = []
        for cls in class_list:
            for idx in self.class_to_indices.get(int(cls), []):
                if idx < self.current_index and idx not in exclude_set:
                    candidate_indices.append(int(idx))
        if len(candidate_indices) == 0:
            return None, None, None
        if len(candidate_indices) > n_samples:
            candidate_indices = np.random.choice(candidate_indices, n_samples, replace=False).tolist()
        idx_tensor = torch.tensor(candidate_indices, device=self.bx.device, dtype=torch.long)
        x_sel = self.bx[idx_tensor]
        y_sel = self.by[idx_tensor]
        if target_device is not None:
            x_sel = x_sel.to(target_device)
            y_sel = y_sel.to(target_device)
        return x_sel, y_sel, idx_tensor.cpu()

    def uncertainty_stats(self):
        if self.current_index == 0:
            self.latest_u_mean = 0.0
            return {'mean': 0.0, 'max': 0.0, 'min': 0.0}
        current_u = self.bu[:self.current_index]
        mean_val = float(current_u.mean().item())
        self.latest_u_mean = mean_val
        return {
            'mean': mean_val,
            'max': float(current_u.max().item()),
            'min': float(current_u.min().item()),
        }

    def mean_uncertainty(self):
        if self.current_index == 0:
            return 0.0
        return float(self.bu[:self.current_index].mean().item())

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
            'evicted_x': None,
            'evicted_y': None,
            'evicted_u': None,
            'evicted_tasks': None,
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
                # 仅顺序填充，无替换；更新索引映射
                with torch.no_grad():
                    add_idx_cpu = added_indices_fill.detach().cpu()
                    add_labels_cpu = y[:offset].detach().cpu()
                    self._update_class_index_maps(
                        add_idx_cpu=add_idx_cpu,
                        add_labels_cpu=add_labels_cpu,
                        remove_idx_cpu=None,
                        remove_labels_cpu=None,
                    )
                return result_info

        self.place_left = False

        x = x[place_left:]
        y = y[place_left:]
        uncertainty = uncertainty[place_left:]

        indices = torch.empty(x.size(0), device=device).uniform_(0, self.n_seen_so_far).long()
        valid_mask = (indices < self.bx.size(0))

        idx_new_data = valid_mask.nonzero().squeeze(-1)
        idx_buffer = indices[idx_new_data]

        # 去重 idx_buffer，避免同一槽位多次替换导致计数/映射错误
        if idx_buffer.numel() > 0:
            unique_buf = []
            unique_src = []
            seen = set()
            for pos, buf_idx in enumerate(idx_buffer.tolist()):
                if buf_idx not in seen:
                    seen.add(buf_idx)
                    unique_buf.append(buf_idx)
                    unique_src.append(idx_new_data[pos].item())
            idx_buffer = torch.tensor(unique_buf, device=device, dtype=torch.long)
            idx_new_data = torch.tensor(unique_src, device=device, dtype=torch.long)

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

        result_info['evicted_x'] = old_samples
        result_info['evicted_y'] = old_labels
        result_info['evicted_u'] = old_uncertainty
        result_info['evicted_tasks'] = old_tasks

        result_info['replaced'] = idx_buffer
        result_info['replaced_src_pos'] = idx_new_data
        result_info['replaced_old_labels'] = old_labels

        if result_info['added'] is None:
            result_info['added'] = idx_buffer
            result_info['added_src_pos'] = idx_new_data
        else:
            result_info['added'] = torch.cat([result_info['added'], idx_buffer], dim=0)
            result_info['added_src_pos'] = torch.cat([result_info['added_src_pos'], idx_new_data], dim=0)

        # 更新类别索引映射
        with torch.no_grad():
            add_idx_cpu = result_info['added'].detach().cpu() if result_info['added'] is not None else None
            add_labels_cpu = self.by[result_info['added']].detach().cpu() if result_info['added'] is not None else None
            self._update_class_index_maps(
                add_idx_cpu=add_idx_cpu,
                add_labels_cpu=add_labels_cpu,
                remove_idx_cpu=idx_buffer.detach().cpu(),
                remove_labels_cpu=old_labels.detach().cpu(),
            )

        return result_info

    def add_priority(self, x, y, t):
        """
        Directly insert samples: fill remaining slots; if full, overwrite random indices.
        This bypasses reservoir sampling to guarantee inclusion.
        """
        n_elem = x.size(0)
        if n_elem == 0:
            return
        device = self.bx.device
        x = x.to(device)
        y = y.to(self.by.device)

        place_left = max(0, self.bx.size(0) - self.current_index)
        if place_left:
            offset = min(place_left, n_elem)
            self.bx[self.current_index: self.current_index + offset].data.copy_(x[:offset])
            self.by[self.current_index: self.current_index + offset].data.copy_(y[:offset])
            self.bt[self.current_index: self.current_index + offset].fill_(t)
            with torch.no_grad():
                add_counts = torch.bincount(y[:offset].view(-1).long().cpu(), minlength=self.num_classes)
                add_counts = add_counts.to(self.class_counts.device)
                self.class_counts[:add_counts.numel()] += add_counts
                add_idx_cpu = torch.arange(self.current_index, self.current_index + offset, dtype=torch.long)
                self._update_class_index_maps(
                    add_idx_cpu=add_idx_cpu,
                    add_labels_cpu=y[:offset].detach().cpu(),
                    remove_idx_cpu=None,
                    remove_labels_cpu=None,
                )
            self.current_index += offset
            self.n_seen_so_far += offset
            if offset == n_elem:
                return
            # still have remaining to insert
            x = x[offset:]
            y = y[offset:]
            n_elem = x.size(0)

        # buffer is full: overwrite random indices
        idx_buffer = torch.randint(low=0, high=self.bx.size(0), size=(n_elem,), device=device)
        # 去重，保留首次出现，确保计数/映射一致
        if idx_buffer.numel() > 0:
            uniq_idx = []
            uniq_pos = []
            seen = set()
            for pos, val in enumerate(idx_buffer.tolist()):
                if val not in seen:
                    seen.add(val)
                    uniq_idx.append(val)
                    uniq_pos.append(pos)
            idx_buffer = torch.tensor(uniq_idx, device=device, dtype=torch.long)
            x = x[uniq_pos]
            y = y[uniq_pos]
        # 维护计数与映射
        with torch.no_grad():
            old_labels = self.by[idx_buffer].clone()
            dec_counts = torch.bincount(old_labels.view(-1).long().cpu(), minlength=self.num_classes)
            dec_counts = dec_counts.to(self.class_counts.device)
            self.class_counts[:dec_counts.numel()] -= dec_counts

        self.bx[idx_buffer] = x
        self.by[idx_buffer] = y
        self.bt[idx_buffer] = t
        with torch.no_grad():
            inc_counts = torch.bincount(y.view(-1).long().cpu(), minlength=self.num_classes)
            inc_counts = inc_counts.to(self.class_counts.device)
            self.class_counts[:inc_counts.numel()] += inc_counts
            self._update_class_index_maps(
                add_idx_cpu=idx_buffer.detach().cpu(),
                add_labels_cpu=y.detach().cpu(),
                remove_idx_cpu=idx_buffer.detach().cpu(),
                remove_labels_cpu=old_labels.detach().cpu(),
            )

    def shuffle_(self):
        n = self.current_index
        if n <= 1:
            return
        perm = torch.randperm(n, device=self.bx.device)
        self.bx[:n] = self.bx[:n][perm]
        self.by[:n] = self.by[:n][perm]
        self.bt[:n] = self.bt[:n][perm]
        self.bu[:n] = self.bu[:n][perm]
        self._rebuild_class_index_maps()

    def delete_up_to(self, remove_after_this_idx):
        keep = max(0, min(self.current_index, remove_after_this_idx))
        self.current_index = keep
        self.n_seen_so_far = max(self.n_seen_so_far, keep)
        # 重算 class_counts
        with torch.no_grad():
            self.class_counts.zero_()
            if keep > 0:
                cnt = torch.bincount(self.by[:keep].view(-1).long().cpu(), minlength=self.num_classes)
                cnt = cnt.to(self.class_counts.device)
                self.class_counts[:cnt.numel()] = cnt
        self._rebuild_class_index_maps()

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

    def sample_by_classes(self, amt, classes=None, exclude_task=None, ret_ind=False):
        """
        从指定类别集合中采样；若未提供类别，则回退到默认采样。
        """
        if classes is None or len(classes) == 0:
            return self.sample(amt, exclude_task=exclude_task, ret_ind=ret_ind)

        if amt <= 0 or self.current_index == 0:
            empty_x = self.bx.new_empty((0,) + self.bx.shape[1:])
            empty_y = self.by.new_empty((0,), dtype=self.by.dtype)
            empty_t = self.bt.new_empty((0,), dtype=self.bt.dtype)
            if ret_ind:
                empty_idx = torch.empty(0, device=self.bx.device, dtype=torch.long)
                return empty_x, empty_y, empty_t, empty_idx
            return empty_x, empty_y, empty_t

        device = self.bx.device
        class_tensor = torch.as_tensor(list(classes), device=device, dtype=self.by.dtype)

        if class_tensor.numel() == 0:
            return self.sample(amt, exclude_task=exclude_task, ret_ind=ret_ind)

        mask = torch.isin(self.by[:self.current_index], class_tensor)
        if exclude_task is not None:
            task_mask = (self.bt[:self.current_index] != exclude_task)
            mask = mask & task_mask

        valid_indices = torch.arange(self.current_index, device=device, dtype=torch.long)[mask]

        if valid_indices.numel() == 0:
            empty_x = self.bx.new_empty((0,) + self.bx.shape[1:])
            empty_y = self.by.new_empty((0,), dtype=self.by.dtype)
            empty_t = self.bt.new_empty((0,), dtype=self.bt.dtype)
            if ret_ind:
                empty_idx = torch.empty(0, device=device, dtype=torch.long)
                return empty_x, empty_y, empty_t, empty_idx
            return empty_x, empty_y, empty_t

        sample_count = min(int(amt), valid_indices.numel())
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

    def print_per_task_num(self):
        _, counts = torch.unique(self.bt, return_counts=True)
        print(f"Number of buffed imgs: {counts.tolist()}")