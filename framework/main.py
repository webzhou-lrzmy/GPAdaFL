#!/usr/bin/env python3
"""
GP-AdaFL 端到端演示
python main.py
"""

import torch, yaml, copy, os, sys
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

# ==============  极简实现  ============== #

def dirichlet_split(dataset, n_clients, alpha=0.1, seed=0):
    """返回 list of indices，每 client 一份"""
    rng = torch.Generator().manual_seed(seed)
    labels = torch.tensor(dataset.targets)
    n_classes = len(dataset.classes)
    label_dist = torch.distributions.Dirichlet(alpha * torch.ones(n_classes)).sample((n_clients,))
    client_idx = [[] for _ in range(n_clients)]
    for cls in range(n_classes):
        cls_idx = (labels == cls).nonzero(as_tuple=True)[0]
        cls_idx = cls_idx[torch.randperm(len(cls_idx), generator=rng)]
        split = (label_dist[:, cls] * len(cls_idx)).int()
        split[-1] = len(cls_idx) - split[:-1].sum()
        pos = 0
        for cid, num in enumerate(split):
            client_idx[cid].extend(cls_idx[pos:pos+num].tolist())
            pos += num
    return client_idx

class SimpleNet(torch.nn.Module):
    def __init__(self, in_dim=28*28, hid=128, n_cls=10):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_dim, hid),
            torch.nn.ReLU(),
            torch.nn.Linear(hid, n_cls)
        )
    def forward(self, x): return self.net(x)

class LocalClient:
    def __init__(self, cid, model, loader, cfg, device):
        self.cid = cid
        self.model = model.to(device)
        self.loader = loader
        self.cfg = cfg
        self.device = device
        self.opt = torch.optim.SGD(self.model.parameters(), lr=cfg['lr'])

    def local_train(self, global_dict, noise_scale):
        self.model.load_state_dict(global_dict)
        self.model.train()
        C = 1.0                                    # 裁剪阈值
        for _ in range(self.cfg['E']):
            for x, y in self.loader:
                x, y = x.to(self.device), y.to(self.device)
                self.opt.zero_grad()
                loss = torch.nn.functional.cross_entropy(self.model(x), y)
                loss.backward()
                # 裁剪 & 噪声
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), C)
                with torch.no_grad():
                    for p in self.model.parameters():
                        p.grad += torch.randn_like(p) * (noise_scale * C)
                self.opt.step()
        return {k: v.cpu() for k, v in self.model.state_dict().items()}

    def grad_norm(self, global_dict):
        self.model.load_state_dict(global_dict)
        total = 0.0
        for p in self.model.parameters():
            param_norm = p.grad.data.norm(2) if p.grad is not None else torch.tensor(0.0)
            total += param_norm.item() ** 2
        return total ** 0.5

class Server:
    def __init__(self, global_model, clients, cfg):
        self.global_model = global_model
        self.clients = clients
        self.cfg = cfg
        self.privacy = 0.0

    def step(self, rnd):
        grads = [c.grad_norm(self.global_model) for c in self.clients]
        avg_norm = sum(grads) / len(grads)
        # 简单动态噪声：norm 越大 → 噪声越大
        noise_scale = 0.5 + 0.5 * torch.sigmoid(torch.tensor(avg_norm - 1.0)).item()

        updates = [c.local_train(self.global_model, noise_scale) for c in self.clients]
        # 均匀聚合
        for key in self.global_model:
            self.global_model[key] = torch.stack([upd[key] for upd in updates]).mean(0)

        # 简化的 RDP 记账（单轮高斯）
        orders = range(2, 65)
        rdp = [(noise_scale ** -2) * alpha / 2 for alpha in orders]
        eps = min(rdp[i] + np.log(1/self.cfg['delta']) / (orders[i]-1) for i in range(len(orders)))
        self.privacy += eps
        return self.privacy <= self.cfg['epsilon_total']

def main():
    os.makedirs('data', exist_ok=True)
    with open('config.yaml', 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 数据
    transform = transforms.ToTensor()
    train_set = datasets.MNIST('data', train=True, download=True, transform=transform)
    client_idx = dirichlet_split(train_set, cfg['K'], alpha=0.1)
    clients = []
    for i, idx in enumerate(client_idx):
        loader = DataLoader(Subset(train_set, idx), batch_size=cfg['B'], shuffle=True)
        model = SimpleNet()
        clients.append(LocalClient(i, model, loader, cfg, device))

    global_model = SimpleNet().state_dict()
    server = Server(global_model, clients, cfg)

    for rnd in tqdm(range(cfg['T']), desc='Rounds'):
        if not server.step(rnd):
            print(f'Budget exhausted at round {rnd}')
            break
    torch.save(global_model, 'gp-adafL_global.pt')
    print('Training finished. Global model saved to gp-adafL_global.pt')

if __name__ == '__main__':
    main()
