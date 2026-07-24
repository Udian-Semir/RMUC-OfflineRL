"""Map-and-entity encoder with masked tactical action heads."""
from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical


class TacticalActorCritic(nn.Module):
    """Scores goal, target and fire-mode from semantic map + state features."""

    def __init__(self, map_channels: int, vector_dim: int, n_goals: int, n_targets: int) -> None:
        super().__init__()
        self.map_encoder = nn.Sequential(
            nn.Conv2d(map_channels, 24, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 4)), nn.Flatten(),
        )
        self.vector_encoder = nn.Sequential(nn.Linear(vector_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.trunk = nn.Sequential(nn.Linear(64 * 3 * 4 + 128, 256), nn.ReLU(), nn.Linear(256, 192), nn.ReLU())
        self.goal_head = nn.Linear(192, n_goals)
        self.target_head = nn.Linear(192, n_targets)
        self.fire_head = nn.Linear(192, 2)
        self.value_head = nn.Linear(192, 1)

    def forward(self, maps: torch.Tensor, vectors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        spatial = self.map_encoder(maps)
        vector = self.vector_encoder(vectors)
        latent = self.trunk(torch.cat((spatial, vector), dim=-1))
        return self.goal_head(latent), self.target_head(latent), self.fire_head(latent), self.value_head(latent).squeeze(-1)

    @staticmethod
    def _masked_distribution(logits: torch.Tensor, mask: torch.Tensor) -> Categorical:
        # At least one valid action is an environment invariant.  Fail early if
        # an integration accidentally violates it instead of sampling a NaN.
        if not torch.all(mask.any(dim=-1)):
            raise RuntimeError("action mask has an all-invalid row")
        return Categorical(logits=logits.masked_fill(~mask.bool(), -1e9))

    def act(
        self,
        maps: torch.Tensor,
        vectors: torch.Tensor,
        goal_mask: torch.Tensor,
        target_mask: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        goal_logits, target_logits, fire_logits, value = self(maps, vectors)
        goal_dist = self._masked_distribution(goal_logits, goal_mask)
        target_dist = self._masked_distribution(target_logits, target_mask)
        fire_dist = Categorical(logits=fire_logits)
        if deterministic:
            actions = torch.stack((goal_dist.probs.argmax(-1), target_dist.probs.argmax(-1), fire_dist.probs.argmax(-1)), dim=-1)
        else:
            actions = torch.stack((goal_dist.sample(), target_dist.sample(), fire_dist.sample()), dim=-1)
        log_prob = goal_dist.log_prob(actions[:, 0]) + target_dist.log_prob(actions[:, 1]) + fire_dist.log_prob(actions[:, 2])
        return actions, log_prob, value

    def evaluate_actions(
        self,
        maps: torch.Tensor,
        vectors: torch.Tensor,
        goal_mask: torch.Tensor,
        target_mask: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        goal_logits, target_logits, fire_logits, value = self(maps, vectors)
        goal_dist = self._masked_distribution(goal_logits, goal_mask)
        target_dist = self._masked_distribution(target_logits, target_mask)
        fire_dist = Categorical(logits=fire_logits)
        log_prob = (goal_dist.log_prob(actions[:, 0]) + target_dist.log_prob(actions[:, 1]) +
                    fire_dist.log_prob(actions[:, 2]))
        entropy = goal_dist.entropy() + target_dist.entropy() + fire_dist.entropy()
        return log_prob, entropy, value
