import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataloader import NuscData, collate_fn
from resnet import resnet18


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        
        # resnet for global map feature
        self.backbone = resnet18()
        
        # encoding the vehicle data
        self.enc1 = nn.Linear(6+512, 16)
        self.enc2 = nn.Linear(16,16)
        
        self.enc3 = nn.Linear(32, 32)
        self.enc4 = nn.Linear(32, 32)
        
        self.mu = nn.Linear(32, 16)
        self.log_var = nn.Linear(32, 16)
        
        # decoding the vehicle data
        self.dec1 = nn.Linear(16, 16)
        self.dec2 = nn.Linear(16, 6)
        
    def get_map_features(self, ids, maps):
        """Returns a tensor that consists of the global map feature within each sample.
        Each map feature is duplicated n times, where n equals the number of vehicles
        in that sample."""
        maps = self.backbone(maps)

        for i in range(batch_size):
            sample_size = (ids == i).sum()
            sample_map = maps[i].repeat(sample_size, 1)
            
            # concatenate the group of sample maps
            if i == 0:
                map_features = sample_map 
            else:
                map_features = torch.cat((map_features, sample_map), dim=0)
        
        return map_features
                
    
    def get_agg(self, x, ids):
        """Returns a tensor that consists of the aggregated data within each sample.
        The aggregated data is duplicated n times, where n equals the number of vehicles
        in that sample."""
        
        for i in range(batch_size):
            sample_size = (ids == i).sum()
            sample_agg = torch.mean(x[ids == i], 0).repeat(sample_size, 1)
            
            # concatenate each group of aggregated data
            if i == 0:
                agg = sample_agg 
            else:
                agg = torch.cat((agg, sample_agg), dim=0)
                
        return agg

    def forward(self, data):
        ids = data.batch_ids
        maps = data.batched_map
        x = data.batched_vehicles
        
        # concatenate global map features
        map_features = self.get_map_features(ids, maps)
        x = torch.cat((x, map_features), 1)
        
        # encode
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        
        # aggregate and concatenate vehicle data within each sample
        agg = self.get_agg(x, ids)
        
        # concatenate the aggregated data with each corresponding sample
        x = torch.cat((x, agg), 1)
        
        # mlp again
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc4(x))
        
        mu = self.mu(x)
        log_var = self.log_var(x)
        
        # reparameterize
        std = torch.exp(0.5*log_var) 
        eps = torch.randn_like(std) 
        sample = mu + (eps * std)
        
        # decode
        
        x = F.relu(self.dec1(sample))
        x = self.dec2(x)
        
        return x, mu, log_var
    
def total_loss(mu, log_var, criterion, output, inputs):
    kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return kl + criterion(output, inputs)

def train(network, inputs, optimizer, criterion):
    optimizer.zero_grad()
    output, mu, log_var = network(inputs)
    loss = total_loss(mu, log_var, criterion, output, inputs.batched_vehicles)
    loss.backward()
    optimizer.step()
    return loss

if __name__ = '__main__':
    train_dataset = NuscData(is_train=True)
    batch_size = 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)
    vehicle_dim = 6
    
    network = VAE().cuda()
    lr = 0.001
    epochs = 50
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        print(f"Epoch {epoch+1} of {epochs}")
        train_epoch_loss = 0

        for i, data in enumerate(train_loader, 0):
            train_epoch_loss += train(network, data.cuda(), optimizer, criterion)

        print(f'Train Loss: {train_epoch_loss.item()/i}')
    
        
