import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv, global_mean_pool


class GraphTransformerEncoder(nn.Module):
    def __init__(self, num_node_tokens, node_emb_dim, hidden_dim, latent_dim, num_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(num_node_tokens, node_emb_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.LeakyReLU(0.2)
        self.convs = nn.ModuleList()

        for i in range(num_layers):
            in_channels = node_emb_dim if i == 0 else hidden_dim
            out_channels = latent_dim if i == num_layers - 1 else hidden_dim
            self.convs.append(TransformerConv(in_channels, out_channels, heads=1, dropout=dropout))

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim if i < num_layers - 1 else latent_dim)
            for i in range(num_layers)
        ])
        self.batch_norm = nn.BatchNorm1d(latent_dim)

    def forward(self, data: Data) -> torch.Tensor:
        x = self.embedding(data.x.view(-1))

        for i, conv in enumerate(self.convs):
            identity = x
            x = conv(x, data.edge_index)
            x = self.layer_norms[i](x)
            x = self.activation(x)
            if i > 0 and identity.shape == x.shape:
                x = x + identity
            x = self.dropout(x)

        out = global_mean_pool(x, data.batch)
        return self.batch_norm(out)


class Generator(nn.Module):
    def __init__(self, noise_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(noise_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)

        self.fc2 = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)

        self.res1_fc1 = nn.Linear(256, 256)
        self.res1_bn1 = nn.BatchNorm1d(256)
        self.res1_act1 = nn.LeakyReLU(0.2, inplace=True)
        self.res1_fc2 = nn.Linear(256, 256)
        self.res1_bn2 = nn.BatchNorm1d(256)
        self.res1_act2 = nn.LeakyReLU(0.2, inplace=True)

        self.res2_fc1 = nn.Linear(256, 256)
        self.res2_bn1 = nn.BatchNorm1d(256)
        self.res2_act1 = nn.LeakyReLU(0.2, inplace=True)
        self.res2_fc2 = nn.Linear(256, 256)
        self.res2_bn2 = nn.BatchNorm1d(256)
        self.res2_act2 = nn.LeakyReLU(0.2, inplace=True)

        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.act3 = nn.LeakyReLU(0.2, inplace=True)
        self.fc_out = nn.Linear(128, output_dim)
        self.act_out = nn.Tanh()
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, a=0.2, nonlinearity="leaky_relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.bn1(self.fc1(noise)))
        x = self.act2(self.bn2(self.fc2(x)))

        residual = x
        x = self.res1_act1(self.res1_bn1(self.res1_fc1(x)))
        x = self.res1_bn2(self.res1_fc2(x)) + residual
        x = self.res1_act2(x)

        residual = x
        x = self.res2_act1(self.res2_bn1(self.res2_fc1(x)))
        x = self.res2_bn2(self.res2_fc2(x)) + residual
        x = self.res2_act2(x)

        x = self.act3(self.bn3(self.fc3(x)))
        return self.act_out(self.fc_out(x))


class GraphCombinedDiscriminator(nn.Module):
    def __init__(
        self,
        num_node_tokens,
        node_emb_dim,
        hidden_dim,
        latent_dim,
        num_layers,
        pos_prior=0.5,
        fc_dropout=0.3,
    ):
        super().__init__()
        self.encoder = GraphTransformerEncoder(
            num_node_tokens, node_emb_dim, hidden_dim, latent_dim, num_layers, dropout=0.3
        )
        self.gradient_reversal_alpha = 0.0
        self.latent_norm = nn.LayerNorm(latent_dim)
        self.input_dropout = nn.Dropout(0.1)

        self.shared_adv = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(latent_dim, 128)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(fc_dropout),
            nn.utils.spectral_norm(nn.Linear(128, 128)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(fc_dropout),
            nn.utils.spectral_norm(nn.Linear(128, 64)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(fc_dropout),
        )
        self.adv_head = nn.utils.spectral_norm(nn.Linear(64, 1))

        self.shared_pu = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(fc_dropout),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(fc_dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(fc_dropout),
        )
        self.pu_head = nn.Linear(64, 1)
        self.init_pu_bias(pos_prior)

    def init_pu_bias(self, pos_prior):
        pos_prior = min(max(float(pos_prior), 1e-4), 1.0 - 1e-4)
        with torch.no_grad():
            bias_value = torch.log(torch.tensor(pos_prior / (1.0 - pos_prior)))
            self.pu_head.bias.data.fill_(bias_value)

    def set_gradient_reversal_alpha(self, alpha):
        self.gradient_reversal_alpha = float(alpha)

    def encode_if_graph(self, x):
        latent = self.encoder(x) if hasattr(x, "batch") else x
        latent = self.latent_norm(latent)
        return self.input_dropout(latent)

    def get_adv_features(self, x) -> torch.Tensor:
        return self.shared_adv(self.encode_if_graph(x))

    def forward(self, x, get_only_pu=False, forward_fake=False):
        latent = self.encode_if_graph(x)
        latent_adv = latent * (1.0 - self.gradient_reversal_alpha) if forward_fake else latent

        if get_only_pu:
            feat_pu = self.shared_pu(latent)
            return None, self.pu_head(feat_pu)

        feat_adv = self.shared_adv(latent_adv)
        feat_pu = self.shared_pu(latent)
        return self.adv_head(feat_adv), self.pu_head(feat_pu)
