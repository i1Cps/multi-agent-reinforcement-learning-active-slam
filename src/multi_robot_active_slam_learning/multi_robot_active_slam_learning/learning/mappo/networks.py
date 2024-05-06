from pathlib import Path
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta


class CriticNetwork(nn.Module):
    def __init__(
        self,
        input_dims: int,
        learning_rate: float,
        fc1: int = 400,
        fc2: int = 300,
        name: str = "critic",
        checkpoint_dir: str = "models/",
        scenario: str = "unclassified",
    ):
        super(CriticNetwork, self).__init__()
        self.checkpoint_dir = Path(checkpoint_dir) / scenario
        self.checkpoint_file = self.checkpoint_dir / (name + "mappo")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.input_dims = input_dims
        self.value_fc1 = fc1
        self.value_fc2 = fc2
        self.fc1 = nn.Linear(input_dims, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.v = nn.Linear(fc2, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state) -> T.Tensor:
        x = T.tanh(self.fc1(state))
        x = T.tanh(self.fc2(x))
        v = self.v(x)
        return v

    def save_checkpoint(self):
        print("... saving checkpoint ...")
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("... loading checkpoint ...")
        self.load_state_dict(T.load(self.checkpoint_file))


class ActorNetwork(nn.Module):
    def __init__(
        self,
        learning_rate: float,
        input_dims: int,
        n_actions: int,
        fc1: int = 256,
        fc2: int = 256,
        name: str = "actor",
        checkpoint_dir: str = "models",
        scenario: str = "unclassified",
    ):
        super(ActorNetwork, self).__init__()
        self.checkpoint_dir = Path(checkpoint_dir) / scenario
        self.checkpoint_file = self.checkpoint_dir / (name + "mappo")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.fc1 = nn.Linear(input_dims, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.alpha = nn.Linear(fc2, n_actions)
        self.beta = nn.Linear(fc2, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state) -> Beta:
        x = T.tanh(self.fc1(state))
        x = T.tanh(self.fc2(x))
        alpha = F.relu(self.alpha(x)) + 1.0
        beta = F.relu(self.beta(x)) + 1.0
        dist = Beta(alpha, beta)
        return dist

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
