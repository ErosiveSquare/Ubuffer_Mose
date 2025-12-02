import torch
from typing import Tuple


class UncertaintyBuffer:
    def __init__(self, capacity: int = 100, device: torch.device = torch.device('cuda')):
        self.capacity = int(capacity)
        self.device = device
        self.x_list = []
        self.y_list = []
        self.u_list = []  # uncertainty scores (higher = harder)

    def __len__(self) -> int:
        return len(self.x_list)

    @torch.no_grad()
    def add_batch(self, x: torch.Tensor, y: torch.Tensor, u: torch.Tensor) -> None:
        """
        Add a batch of candidates into the buffer using replacement policy:
        - If buffer not full, append
        - Else, replace the entry with the smallest u if new u is larger
        Args:
            x: [B, C, H, W]
            y: [B]
            u: [B] (float) uncertainty scores
        """
        if x.numel() == 0:
            return
        x = x.detach().to(self.device)
        y = y.detach().to(self.device)
        u = u.detach().to(self.device).float()

        for i in range(x.size(0)):
            xi = x[i].clone()
            yi = y[i].clone()
            ui = u[i].item()
            if len(self.x_list) < self.capacity:
                self.x_list.append(xi)
                self.y_list.append(yi)
                self.u_list.append(ui)
            else:
                # replace the minimum-uncertainty sample if current is harder
                min_idx = int(torch.tensor(self.u_list).argmin().item())
                if ui > self.u_list[min_idx]:
                    self.x_list[min_idx] = xi
                    self.y_list[min_idx] = yi
                    self.u_list[min_idx] = ui

    @torch.no_grad()
    def sample(self, amt: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample top-k highest-uncertainty samples. If buffer is smaller than amt,
        return all. Returns (x, y, indices_in_buffer)
        """
        if len(self.x_list) == 0:
            empty_x = torch.empty(0, device=self.device)
            empty_y = torch.empty(0, dtype=torch.long, device=self.device)
            empty_ind = torch.empty(0, dtype=torch.long, device=self.device)
            return empty_x, empty_y, empty_ind

        k = min(int(amt), len(self.x_list))
        u_tensor = torch.tensor(self.u_list, device=self.device)
        # topk on uncertainty
        topk_vals, topk_idx = torch.topk(u_tensor, k, largest=True, sorted=False)
        xs = torch.stack([self.x_list[int(i.item())] for i in topk_idx], dim=0)
        ys = torch.stack([self.y_list[int(i.item())] for i in topk_idx], dim=0)
        return xs, ys, topk_idx.long()

    @torch.no_grad()
    def update_uncertainty(self, indices: torch.Tensor, new_u: torch.Tensor) -> None:
        if indices.numel() == 0:
            return
        new_u = new_u.detach().to(self.device).float()
        for pos in range(indices.numel()):
            idx = int(indices[pos].item())
            self.u_list[idx] = float(new_u[pos].item())


