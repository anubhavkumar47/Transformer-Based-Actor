import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerActor(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        embed_dim=128,
        n_heads=4,
        n_layers=2,
        hidden_dim=256,
        dropout_p=0.1
    ):
        super().__init__()
        self.max_action = max_action
        self.state_dim = state_dim
        self.embed_dim = embed_dim

        # Embed each state feature as a token
        self.feature_embed = nn.Linear(1, embed_dim)

        # Learned positional encodings
        self.pos_encoding = nn.Parameter(torch.randn(state_dim, embed_dim))

        # Transformer encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout_p,
            activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Final action projection
        self.project = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, action_dim)
        )

        # Small weight initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -1e-3, 1e-3)
                nn.init.zeros_(m.bias)

    def forward(self, state):
        # state: [batch, state_dim]
        b, s = state.shape
        # reshape to [batch, state_dim, 1] tokens
        x = state.view(b, s, 1)
        # linear embed to [batch, state_dim, embed_dim]
        x = self.feature_embed(x)
        # add positional encodings
        x = x + self.pos_encoding.unsqueeze(0)
        # transformer expects [seq_len, batch, embed_dim]
        x = x.permute(1, 0, 2)
        # pass through transformer
        x = self.transformer(x)
        # back to [batch, seq_len, embed_dim]
        x = x.permute(1, 0, 2)
        # mean-pool over sequence
        x = x.mean(dim=1)
        # project to action_dim
        action = self.project(x)
        # bound actions to [-max_action, max_action]
        action = torch.tanh(action)
        return action * self.max_action