"""SAC RLPD with a paired forward-backward loss on the same replay.

The actor and critic are the quadmanip SAC RLPD trainer without symmetry
augmentation. ``F`` and ``B`` train on the behavior data and do not enter
the actor loss. The prompt ``z`` is the reference embedding stored at rollout.
"""

from __future__ import annotations

import copy
import math
import einops
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Tuple, TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from active_adaptation.envs import _EnvBase

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from hydra.core.config_store import ConfigStore
from tensordict import TensorDict
from tensordict.nn import (
    TensorDictModuleBase,
    TensorDictModule as Mod,
    TensorDictSequential as Seq,
)

from torchrl.data import Composite, TensorSpec
from torchrl.objectives import hold_out_net

import active_adaptation as aa
from active_adaptation.learning.modules import VecNorm, IndependentNormal, CatTensors
from active_adaptation.learning.ppo.common import (
    ACTION_KEY,
    DONE_KEY,
    OBS_KEY,
    CMD_KEY,
    REWARD_KEY,
    TERM_KEY,
    soft_copy_,
)

from active_adaptation.learning.offpolicy.buffer import ReplayBuffer, sample_replay_mix
from active_adaptation.learning.offpolicy.objectives import (
    MultiStepReturn,
    prior_bc_loss,
)
from active_adaptation.learning.offpolicy.reward_normalization import RewardNormalizer
from active_adaptation.learning.utils.opt import MuonAdamWWrapper
from active_adaptation.learning.utils.distributed import (
    unwrap_ddp,
    wrap_ddp,
)
from active_adaptation.utils.profiling import ScopedTimer
# Shared actor / critic trunks / rollout policy with scalar SAC.
from active_adaptation.learning.offpolicy.sac import (
    AlphaModule,
    NormalActor,
    SACRolloutPolicy,
    gaussian_target_entropy,
)

from .critic import collate_reward, make_critic, reward_group_keys, window_outcome
from .fb_paired import (
    FB_KEY,
    BackwardMap,
    ForwardMap,
    project_z,
    uncertainty_target,
)

cs = ConfigStore.instance()


clip_grad_norm_ = nn.utils.clip_grad_norm_


@torch.no_grad()
def weight_norm(module: nn.Module) -> float:
    """L2 norm of linear-map weights in ``module``."""
    total = None
    for sub in module.modules():
        weight = getattr(sub, "weight", None)
        if isinstance(weight, torch.Tensor) and weight.ndim == 2:
            sq = weight.detach().float().square().sum()
            total = sq if total is None else total + sq
    if total is None:
        return 0.0
    return float(total.sqrt())



@dataclass
class SACFBConfig:
    """SAC RLPD on the task reward, plus a paired FB loss that does not train the actor.

    Symmetry augmentation is omitted. ``fb_state`` stays in ``in_keys`` so the
    env keeps that group, and is then removed from the actor input.
    """

    _target_: str = "aa_bfm_learning.sac_fb.SACFBConfig"
    name: str = "sac_fb"
    train_every: int = 4
    buffer_size: int = 2000
    warm_up_steps: int = 200
    lr: float = 5e-4
    # If True, actor/Q use :class:`~active_adaptation.learning.utils.opt.MuonAdamWWrapper` (see ``ppo_symaug``).
    muon: bool = True
    weight_decay: float = 0.02
    # TD learning
    n_steps: int = 3
    gamma: float = 0.99
    utd_ratio: int = 4
    # architecture
    actor_init: str = "zeros"
    # batch sizes
    critic_batch_size: int = 2048
    actor_batch_size: int = 2048
    # target smoothing: this should help Q(s_t, a_t) to generalize locally around a_t
    target_action_noise: float = 0.01
    # AR(1) pre-tanh exploration noise on rollout only: eps_t = rho * eps_{t-1} + sqrt(1-rho^2) * N(0,I).
    use_correlated: bool = True
    # sac specific
    entropy_bonus: float = 1.0
    alpha_init: float = 4e-3
    # If set: H_target = (d/2)*log(2*pi*e*sigma^2) for N(0,sigma^2)^d (FlashSAC).
    # If None: use -dim(A) (common heuristic for tanh-squashed SAC).
    target_entropy_sigma: float | None = 0.15
    soft_bound: float = 2.0 * math.pi

    tau_actor: float = 0.02 # a relatively large value for faster convergence
    tau_Q: float = 0.02  # a relatively large value for faster convergence
    lr_alpha: float = 5e-4
    max_grad_norm: float = 1.0

    debug: bool = False
    vecnorm: bool = True
    # FP16 AMP (CUDA only); separate GradScalers for critic and actor (alpha stays fp32).
    use_amp: bool = True
    # Clamp aggregated rewards at 0 before TD / reward-norm (avoids suicide from negative rewards).
    clamp_reward: bool = False
    # FlashSAC-style: scale learning rewards by running discounted-return stats (buffer stores raw).
    normalize_reward: bool = True
    reward_norm_epsilon: float = 1e-8
    # Ensemble Q (clipped double-Q when ``num_critics=2``).
    num_critics: int = 2
    # c51 | regression | classification. See quadmanip_learning.offpolicy.critic.
    critic: str = "classification"
    # One-hot (continue, terminated, truncated) on the critic trunk. The loss
    # uses the logged window; bootstrap and actor queries use continue.
    critic_outcome: bool = False
    # One critic head per reward group. The actor maximizes the sum.
    group_critic: bool = False

    # path to prior data for RLPD
    prior_data: str | None = None
    # Mix ratio of prior rows inside each update. Distinct from ``prior_fraction``.
    prior_data_ratio: float = 0.4
    # Fraction of prior trajectories to load. 1 keeps the archive. The same
    # seed nests smaller fractions inside larger ones.
    prior_fraction: float = 1.0
    prior_fraction_seed: int = 0
    # Prior floor: use two-hot(G) when MC return-to-go is larger than the online Q value
    prior_floor: bool = False
    # Gated BC toward prior actions when online Q < MC return-to-go (0 disables).
    # NLL (summed over act dims) is typically much larger than mean MSE → smaller default.
    bc_loss: str = "mse"
    bc_coef: float = 0.0
    bc_coef_mse: float = 1.0
    bc_coef_nll: float = 0.05

    # ``fb_state`` is kept for B. It is not part of the SAC observation.
    in_keys: Tuple[str, ...] = (CMD_KEY, OBS_KEY, FB_KEY)
    # Paired FB. The actor still maximizes Q only.
    z_dim: int = 256
    fb_hidden_dim: int = 512
    backward_hidden_dim: int = 256
    num_parallel: int = 2
    lr_f: float = 3.0e-4
    lr_b: float = 1.0e-5
    tau_fb: float = 0.01
    ortho_coef: float = 100.0
    fb_pessimism: float = 0.0
    q_loss_coef: float = 1.0

    def __post_init__(self):
        if self.num_critics < 1:
            raise ValueError(f"num_critics must be >= 1, got {self.num_critics}")
        if self.critic not in ("c51", "regression", "classification"):
            raise ValueError(
                f"critic must be 'c51', 'regression', or 'classification', got {self.critic!r}"
            )
        if self.group_critic and self.clamp_reward:
            raise ValueError(
                "clamp_reward is incompatible with group_critic "
                "(a per-group clamp would zero always-negative groups)."
            )
        if not 0.0 < self.prior_fraction <= 1.0:
            raise ValueError(
                f"prior_fraction must be in (0, 1], got {self.prior_fraction}."
            )
        if self.bc_loss == "mse":
            self.bc_coef = self.bc_coef_mse
        elif self.bc_loss == "nll":
            self.bc_coef = self.bc_coef_nll
        else:
            raise ValueError(f"Unknown bc_loss={self.bc_loss!r}; expected 'mse' or 'nll'.")
        if self.num_parallel < 1:
            raise ValueError(f"num_parallel must be >= 1, got {self.num_parallel}.")
        if not 0.0 < self.tau_fb <= 1.0:
            raise ValueError(f"tau_fb must be in (0, 1], got {self.tau_fb}.")

    def get_class(self):
        return SACFB

cs.store(name="sac_fb", node=SACFBConfig, group="algo")



class SACFB(TensorDictModuleBase):

    def __init__(
        self,
        cfg: SACFBConfig,
        observation_spec: Composite,
        action_spec: Composite,
        reward_spec: TensorSpec,
        device,
        *,
        command_manager,
    ):
        super().__init__()
        if not hasattr(command_manager, "ref_fb_state"):
            raise TypeError("sac_fb requires BFMCommand.ref_fb_state.")
        if FB_KEY not in observation_spec.keys(True, True):
            raise KeyError(f"sac_fb requires an observation group {FB_KEY!r}.")
        if OBS_KEY not in observation_spec.keys(True, True):
            raise KeyError(f"sac_fb requires an observation group {OBS_KEY!r}.")
        # ``fb_state`` is kept on the env by ``in_keys``, then dropped here so
        # the SAC actor and critic stay on command + policy.
        cfg.in_keys = tuple(key for key in cfg.in_keys if key != FB_KEY)
        if not cfg.in_keys:
            raise ValueError("sac_fb actor in_keys is empty after dropping fb_state.")
        self.cfg = cfg
        self.device = device
        self.command_manager = command_manager
        self.observation_spec = observation_spec
        self.action_spec = action_spec
        self.reward_spec = reward_spec
        self.eff_horizon = 1 / (1 - self.cfg.gamma)
        # Replay rows must share these keys. ``in_keys`` is the observation the
        # actor and critic read; a relabeled prior uses ``command_task`` here
        # instead of the teacher ``command``.
        obs_keys = tuple(self.cfg.in_keys)
        self.train_keys = (
            *obs_keys,
            *[("next", key) for key in obs_keys],
            ACTION_KEY,
            REWARD_KEY,
            TERM_KEY,
            DONE_KEY,
            ("next", "discount"),
            "is_init",
        )

        self._distributed = aa.is_distributed()
        if self._distributed and not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError(
                "Distributed training is enabled but torch.distributed is not initialized."
            )

        fake = observation_spec.zero()
        spec_keys = set(observation_spec.keys(True, True))
        missing = [key for key in self.cfg.in_keys if key not in spec_keys]
        if missing:
            raise ValueError(
                f"algo.in_keys {list(self.cfg.in_keys)} is missing {missing} "
                f"from the observation spec {sorted(observation_spec.keys())}."
            )
        obs_dim = sum(int(fake[key].shape[-1]) for key in self.cfg.in_keys)
        preproc = []
        if len(self.cfg.in_keys) == 1:
            preproc.append(Mod(nn.Identity(), [self.cfg.in_keys[0]], ["_input"]))
        else:
            preproc.append(CatTensors(list(self.cfg.in_keys), "_input", del_keys=False, sort=False))
        self.act_dim = action_spec.shape[-1]
        # since entropy grows linearly with the action dimension, we need to scale it down
        self.entropy_scale = 1.0 / math.sqrt(self.act_dim)

        if self.cfg.vecnorm:
            self.vecnorm_obs = VecNorm(obs_dim).to(device)
        else:
            self.vecnorm_obs = nn.Identity()
        preproc.append(Mod(self.vecnorm_obs, ["_input"], ["_input_normed"]))
        self.preproc = Seq(*preproc).to(device)

        if self.cfg.group_critic:
            self.reward_group_keys = reward_group_keys(reward_spec)
        else:
            self.reward_group_keys = []
        self.num_groups = max(len(self.reward_group_keys), 1)
        if self.cfg.normalize_reward:
            # Std-normalized returns are O(1); fixed atom support (not task-tuned).
            v_min, v_max, num_atoms = -0.5, 5.0, 101
        else:
            v_min, v_max = -1.0, 9.0
            num_atoms = int((v_max - v_min) / 0.05) + 1
        self.Q = make_critic(
            self.cfg.critic,
            obs_dim,
            self.act_dim,
            num_atoms=num_atoms,
            v_min=v_min,
            v_max=v_max,
            num_critics=self.cfg.num_critics,
            outcome=self.cfg.critic_outcome,
            num_groups=self.num_groups,
        ).to(device)

        self.DistClass = IndependentNormal
        self.actor = NormalActor(
            obs_dim,
            self.act_dim,
            std_max=1.0,
            std_min=0.001,
            action_init=self.cfg.actor_init,
        ).to(device)

        self.Q_target = copy.deepcopy(self.Q).to(device)
        self.actor_target = copy.deepcopy(self.actor).to(device)
        self.Q_target.requires_grad_(False)
        self.Q_target.eval()
        self.actor_target.requires_grad_(False)
        self.actor_target.eval()

        if self.cfg.target_entropy_sigma is not None:
            self.target_entropy = gaussian_target_entropy(
                self.act_dim, self.cfg.target_entropy_sigma
            )
        else:
            self.target_entropy = - 0.5 * float(self.act_dim)
        self.alpha = AlphaModule(self.cfg.alpha_init).to(device)
        self.opt_alpha = torch.optim.Adam(self.alpha.parameters(), lr=self.cfg.lr_alpha)
        if self.cfg.muon:
            self.opt_actor = MuonAdamWWrapper(
                [self.actor],
                lr=self.cfg.lr,
                weight_decay=self.cfg.weight_decay,
            )
            self.opt_Q = MuonAdamWWrapper(
                [self.Q],
                lr=self.cfg.lr,
                weight_decay=self.cfg.weight_decay,
            )
        else:
            self.opt_actor = torch.optim.AdamW(self.actor.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
            self.opt_Q = torch.optim.AdamW(self.Q.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

        self.global_step = 0

        self.msr = (
            MultiStepReturn(self.cfg.gamma, self.cfg.n_steps).to(device)
            if self.cfg.n_steps > 1
            else None
        )

        self.reward_normalizer: RewardNormalizer | None = None
        if self.cfg.normalize_reward:
            self.reward_normalizer = RewardNormalizer(
                gamma=float(self.cfg.gamma),
                load_rms=False,
                device=self.device if isinstance(self.device, torch.device) else torch.device(self.device),
                epsilon=float(self.cfg.reward_norm_epsilon),
                reward_dim=self.num_groups,
            )

        # F and B train on the behavior data. They do not enter the actor loss.
        fake_fb = observation_spec.zero()
        self.fb_obs_dim = int(fake_fb[OBS_KEY].shape[-1])
        self.b_dim = int(fake_fb[FB_KEY].shape[-1])
        self.z_dim = int(cfg.z_dim)
        self.vecnorm_fb_obs = VecNorm(self.fb_obs_dim).to(device)
        self.vecnorm_b = VecNorm(self.b_dim).to(device)
        self.B = BackwardMap(self.b_dim, self.z_dim, cfg.backward_hidden_dim).to(device)
        self.F = ForwardMap(
            self.fb_obs_dim, self.act_dim, self.z_dim, cfg.fb_hidden_dim, cfg.num_parallel
        ).to(device)
        self.B_target = copy.deepcopy(self.B).requires_grad_(False).eval()
        self.F_target = copy.deepcopy(self.F).requires_grad_(False).eval()
        self.opt_B = torch.optim.Adam(self.B.parameters(), lr=cfg.lr_b)
        self.opt_F = torch.optim.Adam(self.F.parameters(), lr=cfg.lr_f)

        # Distributed wiring: wrap *after* deepcopy of targets so target nets
        # stay plain modules, and *after* ``alpha`` / optimizers exist so
        # the initial broadcast includes them. DDP shares the underlying
        # parameter tensors with the wrapped module, so optimizers built from
        # ``self.actor.parameters()`` keep pointing at the same params.
        if self._distributed:
            self._wrap_ddp(local_rank=aa.get_local_rank())
            self._broadcast_parameters()

        _dev = torch.device(device) if not isinstance(device, torch.device) else device
        self._amp_device_type = _dev.type
        self._amp_enabled = bool(self.cfg.use_amp and _dev.type == "cuda")
        # Separate scalers so critic/actor loss scales and Inf/NaN skips stay independent.
        self.grad_scaler_Q = GradScaler(self._amp_device_type, enabled=self._amp_enabled)
        self.grad_scaler_actor = GradScaler(self._amp_device_type, enabled=self._amp_enabled)
        # self.compute_target = torch.compile(
        #     self._compute_target,
        #     mode="reduce-overhead",
        # ) # compiling sometimes give worse performance, use with caution
        self.compute_target = self._compute_target

        group_keys = self.reward_group_keys if self.cfg.group_critic else None

        def fn(rew: torch.Tensor | TensorDict) -> torch.Tensor:
            return collate_reward(rew, group_keys=group_keys, clamp=self.cfg.clamp_reward)
        self.reward_collate_fn = fn

    def _autocast(self):
        if not self._amp_enabled:
            return nullcontext()
        return autocast(
            device_type=self._amp_device_type,
            dtype=torch.float16,
        )

    def _wrap_ddp(self, local_rank: int) -> None:
        device = (
            torch.device(self.device)
            if not isinstance(self.device, torch.device)
            else self.device
        )
        ddp_kwargs: dict[str, Any] = {
            "broadcast_buffers": True,
            "find_unused_parameters": False,
        }
        if device.type == "cuda":
            ddp_kwargs.update(device_ids=[local_rank], output_device=local_rank)
        self.actor = wrap_ddp(self.actor, **ddp_kwargs)
        self.Q = wrap_ddp(self.Q, **ddp_kwargs)
        self.alpha = wrap_ddp(self.alpha, **ddp_kwargs)
        self.F = wrap_ddp(self.F, **ddp_kwargs)
        self.B = wrap_ddp(self.B, **ddp_kwargs)

    @torch.no_grad()
    def _broadcast_parameters(self) -> None:
        """Make rank-0's parameters/buffers the source of truth at startup.

        Includes the target networks (deepcopied locally, so their initial RNG
        state would otherwise diverge across ranks), :attr:`vecnorm_obs`, and
        :attr:`alpha`.
        """
        if not self._distributed:
            return
        for module in (
            self.vecnorm_obs,
            self.actor,
            self.actor_target,
            self.Q,
            self.Q_target,
            self.alpha,
            self.F,
            self.F_target,
            self.B,
            self.B_target,
            self.vecnorm_fb_obs,
            self.vecnorm_b,
        ):
            for param in module.parameters():
                dist.broadcast(param.data, src=0)
            for buffer in module.buffers():
                dist.broadcast(buffer.data, src=0)

    def make_tensordict_primer(self):
        """Register correlated-noise state **before** constructing :class:`SAC` so replay ``fake_tensordict`` matches rollouts."""
        from torchrl.envs import TensorDictPrimer
        from torchrl.data import UnboundedContinuous, BoundedContinuous, Composite

        shape = tuple(self.action_spec.shape)
        dev = torch.device(self.device)
        spec = {
            "prev_noise": UnboundedContinuous(shape, device=dev),
            "rho": BoundedContinuous(low=0.0, high=1.0, shape=[shape[0], 1], device=dev)
        }
        return TensorDictPrimer(
            Composite(spec, shape=[shape[0]], device=dev),
            random=self.cfg.use_correlated,
            reset_key="done",
            expand_specs=False,
        )
    
    @classmethod
    def from_env(cls, cfg: SACFBConfig, env: _EnvBase, device: torch.device):
        return cls(
            cfg=cfg,
            observation_spec=env.observation_spec,
            action_spec=env.action_spec,
            reward_spec=env.reward_spec,
            device=device,
            command_manager=env.command_manager,
        )

    def get_rollout_policy(self, mode: str = "train", critic: bool = False) -> TensorDictModuleBase:
        """Train: optional AR(1) pre-tanh rollout noise; eval/deploy: deterministic squash of the Gaussian mean."""
        policy = SACRolloutPolicy(
            self.preproc if mode == "train" else VecNorm.freeze()(self.preproc),
            self.actor,
            self.DistClass,
            use_correlated=self.cfg.use_correlated,
            Q=self.Q if critic else None,
            reward_normalizer=self.reward_normalizer,
            critic=critic,
        )
        return SACFBRolloutPolicy(policy, self, eval_mode=mode == "eval")

    def on_stage_start(self, stage: str, env: _EnvBase):
        # we will not create buffer when not training
        fake_rb = (
            env.fake_tensordict()
            .exclude(("next", "stats"), "collector")
        )
        fake_rb["loc"] = torch.zeros(fake_rb.shape[0], self.actor.act_dim)
        fake_rb["z"] = torch.zeros(*fake_rb.shape, self.z_dim, device=fake_rb.device)
        observation_keys = set(env.observation_spec.keys(True, True))
        observation_keys.add("z")
        observation_keys = observation_keys - {"prev_noise", "rho"}
        self.rb = ReplayBuffer.from_fake(
            self.cfg.buffer_size, fake_rb,
            fake_bootstrap=True,
            observation_keys=list(observation_keys),
        )
        print("Primary buffer:")
        print(self.rb)
        if self.cfg.prior_data is not None:
            self.rb_prior = ReplayBuffer.from_rollout(
                self.cfg.prior_data,
                fake_bootstrap=True,
                observation_keys=list(observation_keys),
            )
            # Drop rollout-only fields before the H2D copy, then keep the ring
            # on the training device so samples are not gathered on CPU.
            self.rb_prior.retain_present(self.train_keys)
            self.rb_prior.subsample_envs(
                self.cfg.prior_fraction, seed=self.cfg.prior_fraction_seed,
            )
            self.rb_prior.to(self.device)
            self.rb_prior.compute_return(gamma=self.cfg.gamma, reward_collate_fn=self.reward_collate_fn)
            print("Prior data buffer:")
            print(self.rb_prior)
        else:
            self.rb_prior = None

        # ``self.Q`` / ``self.actor`` may be DDP-wrapped; strip the wrapper so
        # the state dict's keys match the (plain) target nets.
        self.Q_target.load_state_dict(unwrap_ddp(self.Q).state_dict())
        self.actor_target.load_state_dict(unwrap_ddp(self.actor).state_dict())
        self.F_target.load_state_dict(unwrap_ddp(self.F).state_dict())
        self.B_target.load_state_dict(unwrap_ddp(self.B).state_dict())

    def step(self, tensordict: TensorDict):
        """For off-policy algorithms, which typically update more frequently than
        on-policy algorithms, we do not use a collector to collect stacked transitions.

        Instead, we directly push the collected transition to the replay buffer.
        """

        self.global_step += 1
        td = tensordict.exclude(("next", "stats"), "collector")
        
        if self.reward_normalizer is not None:
            self.reward_normalizer.update_reward_stats(
                reward=self.reward_collate_fn(td[REWARD_KEY]),
                terminated=td[TERM_KEY],
                truncated=td["next", "truncated"],
            )
        self.rb.push(td)

        if self.global_step > self.cfg.warm_up_steps and self.global_step % self.cfg.train_every == 0:
            return self.train_op()
        else:
            return {}
    
    def maybe_reset(self, next_tensordict: TensorDict):
        done = next_tensordict["done"]
        if done.any():
            pass
        return next_tensordict

    @ScopedTimer("sac_train")
    @VecNorm.freeze()
    def train_op(self):
        # Sync per-rank running stats *before* any consumer (UTD loop /
        # actor update) reads them. VecNorm updates happen during rollouts
        # (outside ``train_op``) and the reward normalizer was just updated
        # above, so both are at the latest-but-divergent state across ranks.
        with torch.no_grad():
            if self._distributed and self.cfg.vecnorm:
                self.vecnorm_obs.synchronize(mode="broadcast")
            if self._distributed:
                self.vecnorm_fb_obs.synchronize(mode="broadcast")
                self.vecnorm_b.synchronize(mode="broadcast")
            if self._distributed and self.cfg.normalize_reward:
                self.reward_normalizer.synchronize(mode="broadcast")

        infos: dict = {"rb_size": len(self.rb)}

        critic_iters = self.cfg.train_every * self.cfg.utd_ratio
        for i in range(critic_iters):
            batch, batch_prior = sample_replay_mix(
                self.rb,
                self.rb_prior,
                self.cfg.critic_batch_size,
                self.cfg.prior_data_ratio,
                steps=self.cfg.n_steps,
                next_obs=True,
                device=self.device,
            )
            d = i == critic_iters - 1
            # One-step FB on the online row. Prior rows have no reference ``z``.
            fb_step = batch[0] if batch.batch_dims >= 2 else batch
            fb_info = self._update_fb(fb_step.clone())
            info = self.train_critic(batch, batch_prior=batch_prior, diagnostics=d)
        infos.update(info)
        infos.update(fb_info)
        soft_copy_(unwrap_ddp(self.F), self.F_target, tau=self.cfg.tau_fb)
        soft_copy_(unwrap_ddp(self.B), self.B_target, tau=self.cfg.tau_fb)

        # actor update is delayed and has fewer iterations
        actor_iters = self.cfg.train_every
        for j in range(actor_iters):
            d = j == actor_iters - 1
            info = self.train_actor(diagnostics=d)
        infos.update(info)

        infos["actor/weight_norm"] = weight_norm(self.actor)
        infos["critic/weight_norm"] = weight_norm(self.Q)

        return dict(sorted(infos.items()))

    @ScopedTimer("train_critic")
    def train_critic(
        self,
        batch: TensorDict,
        batch_prior: TensorDict | None = None,
        diagnostics: bool = False,
    ):
        self.Q.train()
        batch = batch.select(*self.train_keys, inplace=True, strict=False)
        B_online = batch.shape[1]

        G_prior: torch.Tensor | None = None
        if batch_prior is not None:
            # MC return-to-go for Cal-QL-style prior Q floor (dropped by train_keys select).
            G_prior = batch_prior["G"]
            if len(batch_prior.shape) == 2:
                G_prior = G_prior[0]  # [steps, B, 1] -> [B, 1]
            batch_prior = batch_prior.select(*self.train_keys, inplace=True, strict=False)
            B_prior = batch_prior.shape[1]
            batch = torch.cat([batch, batch_prior], dim=1)
        else:
            B_prior = 0
        B_eff = B_online + B_prior

        reward = self.reward_collate_fn(batch[REWARD_KEY])

        if self.cfg.debug:
            reward = torch.ones_like(reward) * (1.0 - self.cfg.gamma)

        if self.reward_normalizer is not None:
            reward = self.reward_normalizer.normalize_rewards(reward)
        else:
            # scale by effective horizon
            reward = reward * (1.0 - self.cfg.gamma)

        # maybe concat and normalize the observation
        self.preproc(batch)
        self.preproc(batch["next"])

        if self.cfg.n_steps == 1:
            obs = batch["_input_normed"]
            act = batch[ACTION_KEY]
            next_obs = batch["next", "_input_normed"]
            term = batch[TERM_KEY].float()
            env_disc = batch.get(("next", "discount"))
            if env_disc is None:
                env_disc = torch.ones_like(term)
            discount = self.cfg.gamma * env_disc * (1.0 - term)
            is_init = batch["is_init"]
            term_flat = batch[TERM_KEY]
            if term_flat.dim() > 1 and term_flat.shape[-1] == 1:
                term_flat = term_flat.squeeze(-1)
            terminated = term_flat.bool()
        else:
            assert self.msr is not None
            obs = batch["_input_normed"][0]
            act_n = batch[ACTION_KEY]
            env_disc_ms = batch.get(("next", "discount"))
            if env_disc_ms is not None:
                env_disc_ms = env_disc_ms[: self.msr.n_steps]
            act_n, next_obs, reward, discount, terminated = self.msr(
                actions=act_n,
                next_observations=batch["next", "_input_normed"],
                rewards=reward[:self.msr.n_steps],
                terminated=batch[TERM_KEY],
                done=batch[DONE_KEY],
                env_discount=env_disc_ms,
            )
            act = act_n[:, 0]
            is_init = batch["is_init"][0]

        outcome = None
        if self.cfg.critic_outcome:
            outcome = window_outcome(batch[TERM_KEY], batch[DONE_KEY], self.cfg.n_steps)

        prior_floor_frac: float | None = None
        with self._autocast():
            with ScopedTimer("compute_target"):
                q_target, q_target_ev = self.compute_target(next_obs, reward, discount)

        # Prior floor in fp32: y = max(G, E[y_TD]); project(G) when MC wins.
        if G_prior is not None and B_prior > 0 and self.cfg.prior_floor:
            prior_td = q_target[B_online:B_eff].float()
            td_ev = q_target_ev[B_online:B_eff].float()
            if self.reward_normalizer is not None:
                g_q = self.reward_normalizer.normalize_rewards(G_prior)
            else:
                g_q = G_prior * (1.0 - self.cfg.gamma)
            g_q = g_q.float().reshape_as(td_ev)
            use_mc = g_q > td_ev
            prior_floor_frac = use_mc.float().mean().item()
            p_mc = self.Q_target.project(g_q)
            q_target = q_target.float().clone()
            q_target[B_online:B_eff] = torch.where(use_mc, p_mc, prior_td)

        with self._autocast():
            # as of torch 2.11, compiling loss computation leads to numerically
            # inconsistent results and degrades performance

            pred = self.Q(obs, act, outcome)
            per_sample_q_loss = self.Q.compute_loss(pred, q_target)
            valid = (1.0 - is_init.float()).reshape_as(per_sample_q_loss)
            denom = valid.sum().clamp_min(1e-8)
            q_loss = (per_sample_q_loss * valid).sum() / denom

        self.opt_Q.zero_grad(set_to_none=True)
        if self._amp_enabled:
            self.grad_scaler_Q.scale(q_loss).backward()
            # Must unscale before all-reduce / clip / grad norm: those are only
            # meaningful on the physical (unscaled) gradients; grad_scaler.step
            # still runs Inf/NaN checks afterwards.
            self.grad_scaler_Q.unscale_(self.opt_Q)
            critic_grad_norm = clip_grad_norm_(
                self.Q.parameters(), max_norm=self.cfg.max_grad_norm
            )
            self.grad_scaler_Q.step(self.opt_Q)
            self.grad_scaler_Q.update()
        else:
            q_loss.backward()
            critic_grad_norm = clip_grad_norm_(self.Q.parameters(), max_norm=self.cfg.max_grad_norm)
            self.opt_Q.step()

        soft_copy_(self.Q, self.Q_target, tau=self.cfg.tau_Q)

        if not diagnostics:
            return

        infos: dict = {
            "critic/q_loss": q_loss.item(),
            "critic/grad_norm": critic_grad_norm.item(),
        }
        with torch.no_grad():
            if self.cfg.n_steps > 1:
                # Online-only policy/action mismatch at t+1 (exclude prior data).
                obs_t1 = batch["_input_normed"][1, :B_online]
                act_t1 = batch[ACTION_KEY][1, :B_online]
                done_t0 = batch[DONE_KEY][0, :B_online].reshape(B_online)
                alive_t1 = ~done_t0.bool()
                if alive_t1.any():
                    policy_act_t1 = self.actor(obs_t1)[0]
                    l2_t1 = torch.linalg.vector_norm(policy_act_t1 - act_t1, dim=-1)
                    infos["critic/action_mismatch_t1"] = l2_t1[alive_t1].mean().item()

            logits = self.Q(obs, act, outcome)
            q = self.Q.expected_values(logits)
            q_lower = self.Q.expected_values(logits, risk_alpha=0.5)
            q_upper = self.Q.expected_values(logits, risk_alpha=-0.5)
            infos.update(self.Q.support_saturation_stats(logits))

            # Q is trained on normalized rewards when reward_normalizer is active;
            # map logs to PPO effective-horizon scale (``* S * (1 - gamma)``).
            if self.reward_normalizer is not None:
                q = self.reward_normalizer.denormalize_return_values(
                    q, num_critics=self.cfg.num_critics
                )
                q_lower = self.reward_normalizer.denormalize_return_values(
                    q_lower, num_critics=self.cfg.num_critics
                )
                q_upper = self.reward_normalizer.denormalize_return_values(
                    q_upper, num_critics=self.cfg.num_critics
                )

            if self.num_groups > 1:
                for i, name in enumerate(self.reward_group_keys):
                    infos[f"critic/q_value/{name}"] = q[:B_online, ..., i].mean().item()
                    if B_prior > 0:
                        infos[f"critic/prior_q_value/{name}"] = q[B_online:B_eff, ..., i].mean().item()
                q = q.sum(dim=-1)
                q_lower = q_lower.sum(dim=-1)
                q_upper = q_upper.sum(dim=-1)

            infos["critic/q_lower"] = q_lower.mean().item()
            infos["critic/q_upper"] = q_upper.mean().item()

            # online statistics
            q_val_mean = q[:B_online].mean().item()
            q_val_max = q[:B_online].max().item()
            q_val_std = q[:B_online].std(dim=-1).mean().item()

        infos["critic/q_value"] = q_val_mean
        infos["critic/q_max"] = q_val_max
        infos["critic/q_std"] = q_val_std
        
        if B_prior > 0:
            q_prior = q[B_online: B_eff]
            infos["critic/prior_q_value"] = q_prior.mean().item()
            infos["critic/prior_q_max"] = q_prior.max().item()
            if prior_floor_frac is not None:
                infos["critic/prior_return_floor_frac"] = prior_floor_frac

        if terminated.any():
            q_val_terminated = q[terminated.reshape(q.shape[0])]
            infos["critic/q_value_terminated"] = q_val_terminated.mean().item()
            infos["critic/q_loss_terminated"] = per_sample_q_loss[terminated.reshape(q.shape[0])].mean().item()

        return infos

    @torch.no_grad()
    def _compute_target(
        self, next_obs: torch.Tensor, reward: torch.Tensor, discount: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # actions are sampled with uncorrelated noise
        loc, scale = self.actor_target(next_obs)
        dist = self.DistClass(loc, scale)
        next_action = dist.sample()

        next_log_prob = dist.log_prob(next_action)
        alpha = self.alpha()
        # One entropy bonus on the summed Q. Split it across groups so each
        # backup stays r_g + γ Q_g and the sum still matches scalar SAC.
        lp = next_log_prob.reshape(reward.shape[0], 1)
        entropy_bonus = (-alpha * lp) * self.entropy_scale / self.num_groups
        discount_b = discount.reshape(reward.shape[0], 1)
        adjusted_reward = reward + discount_b * self.cfg.entropy_bonus * entropy_bonus
        noise = torch.randn_like(next_action).clamp(-3.0, 3.0)
        return self.Q_target.compute_target(
            next_obs,
            next_action + noise * self.cfg.target_action_noise,
            adjusted_reward,
            discount,
        )

    @ScopedTimer("train_actor")
    def train_actor(self, diagnostics: bool = False):
        # actor training does not need next observations
        batch, batch_prior = sample_replay_mix(
            self.rb,
            self.rb_prior,
            self.cfg.actor_batch_size,
            self.cfg.prior_data_ratio,
            steps=1,
            next_obs=False,
            device=self.device,
        )
        batch = batch.select(*self.train_keys, strict=False) # [N,]
        B_online = batch.shape[0]
        if batch_prior is not None:
            B_prior = batch_prior.shape[0]
            G = batch_prior["G"]
            steps_to_go = batch_prior["steps_to_go"]
            batch_prior = batch_prior.select(*self.train_keys, strict=False)
            batch = torch.cat([batch, batch_prior], dim=0)
        else:
            B_prior = 0

        self.preproc(batch)
        obs = batch["_input_normed"]
        act = batch[ACTION_KEY]
        is_init = batch["is_init"]

        with hold_out_net(self.Q), self._autocast():
            loc, scale = self.actor(obs)
            dist = self.DistClass(loc, scale)
            action_update = dist.rsample((4,))  # [4, N, D]
            entropy_est = -dist.log_prob(action_update).mean(dim=0)
            # Per-ensemble-head expected values [N, K, C]; actor uses the mean head.
            q_grouped = self.Q.get_values(
                obs,
                einops.rearrange(action_update, "k n d -> n k d"),
            )
            q_heads = self.Q.sum_groups(q_grouped)
            q = q_heads.mean(dim=-1)
            policy_term = -q.mean(dim=1)  # average over K action samples

            bc_term = torch.zeros_like(policy_term)
            bc_gate = None
            advantage = None
            group_advantage = None
            horizon_mask = None
            if self.rb_prior is not None:
                # Gate BC when π loses to demo MC return on long horizons.
                # Q is denormalized to the same (1-γ) log scale as G.
                prior_sl = slice(B_online, B_online + B_prior)
                horizon_mask = steps_to_go > self.eff_horizon
                q_prior = q.detach()[prior_sl].mean(dim=-1, keepdim=True)
                if self.reward_normalizer is not None and self.num_groups > 1:
                    q_den = self.reward_normalizer.denormalize_return_values(
                        q_grouped.detach(), num_critics=self.cfg.num_critics
                    )
                    q_prior = q_den.sum(dim=-1)[prior_sl].mean(dim=(1, 2)).unsqueeze(-1)
                elif self.reward_normalizer is not None:
                    q_prior = self.reward_normalizer.denormalize_return_values(q_prior)
                if self.num_groups > 1:
                    G_log = G.sum(dim=-1, keepdim=True) * (1.0 - self.cfg.gamma)
                    if self.reward_normalizer is not None:
                        q_g = self.reward_normalizer.denormalize_return_values(
                            q_grouped.detach(), num_critics=self.cfg.num_critics
                        )
                    else:
                        q_g = q_grouped.detach()
                    q_g = q_g[prior_sl].mean(dim=(1, 2))
                    G_g = (G * (1.0 - self.cfg.gamma)).reshape(q_g.shape[0], self.num_groups)
                    group_advantage = q_g - G_g
                else:
                    G_log = G * (1.0 - self.cfg.gamma)
                assert q_prior.shape == G_log.shape, f"{q_prior.shape} != {G_log.shape}"
                advantage = q_prior - G_log
                bc_gate = torch.relu(-advantage).clamp_max(1.0) * horizon_mask.float()
                prior_dist = self.DistClass(loc[prior_sl], scale[prior_sl])
                bc = prior_bc_loss(
                    self.cfg.bc_loss,
                    action_pred=action_update[:, prior_sl],
                    action_demo=batch_prior[ACTION_KEY],
                    dist=prior_dist,
                )
                bc_term[prior_sl] = bc * bc_gate.squeeze(-1)

        alpha = self.alpha()
        actor_loss = (
            policy_term
            + alpha.detach() * (-entropy_est.reshape_as(policy_term) * self.entropy_scale)
            + 0.01 * ((loc/self.cfg.soft_bound)**6).sum(-1).reshape_as(policy_term)
            + self.cfg.bc_coef * bc_term
        )
        valid = (1.0 - is_init.float()).reshape_as(actor_loss)
        denom = valid.sum().clamp_min(1.0)
        actor_loss = (actor_loss * valid).sum() / denom

        q_action_grad_norm: torch.Tensor | None = None
        q_action_grad_disagreement: torch.Tensor | None = None
        if diagnostics:
            # Per-critic ∇_a Q(s, a). Norm from ensemble-mean grad; disagreement
            # is pairwise (1 - mean_{i<j} cos) so comparable across ensemble sizes.
            grads = []
            for c in range(q_heads.shape[-1]):
                (grad_c,) = torch.autograd.grad(
                    q_heads[..., c].sum(),
                    action_update,
                    retain_graph=True,
                    create_graph=False,
                )
                grads.append(grad_c)
            grads = torch.stack(grads, dim=0)  # [C, K, N, D]
            q_action_grad_norm = grads.mean(dim=0).norm(dim=-1).mean()

            eps = 1e-8
            num_critics = grads.shape[0]
            g_norm = grads.norm(dim=-1, keepdim=True)
            g_unit = grads / g_norm.clamp_min(eps)
            # [C, C, K, N]
            cos = torch.einsum("iknd,jknd->ijkn", g_unit, g_unit)
            valid = (g_norm.squeeze(-1) > eps)  # [C, K, N]
            valid_pair = valid.unsqueeze(0) & valid.unsqueeze(1)
            tril = torch.ones(
                num_critics,
                num_critics,
                device=grads.device,
                dtype=torch.bool,
            ).tril(diagonal=-1)
            pair_mask = tril.view(num_critics, num_critics, 1, 1) & valid_pair
            if pair_mask.any():
                q_action_grad_disagreement = 1.0 - cos[pair_mask].mean()
            else:
                q_action_grad_disagreement = grads.new_zeros(())

        self.opt_alpha.zero_grad(set_to_none=True)
        alpha_loss = -(alpha * (-entropy_est.detach() + self.target_entropy)).mean()
        alpha_loss.backward()
        self.opt_alpha.step()

        self.opt_actor.zero_grad(set_to_none=True)
        if self._amp_enabled:
            self.grad_scaler_actor.scale(actor_loss).backward()
            self.grad_scaler_actor.unscale_(self.opt_actor)
            actor_grad_norm = nn.utils.clip_grad_norm_(
                self.actor.parameters(), max_norm=self.cfg.max_grad_norm
            )
            self.grad_scaler_actor.step(self.opt_actor)
            self.grad_scaler_actor.update()
        else:
            actor_loss.backward()
            actor_grad_norm = nn.utils.clip_grad_norm_(
                self.actor.parameters(), max_norm=self.cfg.max_grad_norm
            )
            self.opt_actor.step()
        soft_copy_(self.actor, self.actor_target, tau=self.cfg.tau_actor)

        if not diagnostics:
            return 

        assert q_action_grad_norm is not None
        assert q_action_grad_disagreement is not None
        with torch.no_grad():
            # q is already the mean head of the summed groups, shape [N, K].
            # Group scales are [G], so denormalize the per-group values first.
            q_for_log = q_grouped.detach()
            if self.reward_normalizer is not None:
                q_for_log = self.reward_normalizer.denormalize_return_values(
                    q_for_log,
                    num_critics=self.cfg.num_critics if self.num_groups > 1 else None,
                )
            q_for_log = self.Q.sum_groups(q_for_log).mean(dim=-1)
            # prior data may not contain "loc" key
            # mean_change = (dist.loc[: batch.shape[0]] - batch["loc"]).abs().mean()
            infos = {
                "actor/loss": actor_loss.item(),
                "actor/grad_norm": actor_grad_norm.item(),
                "actor/alpha": alpha.item(),
                "actor/entropy": entropy_est.mean().item(),
                # "actor/mean_change": mean_change.item(),
                "actor/q_std": q_for_log.std(dim=1).mean().item(),
                "actor/q_action_grad_norm": q_action_grad_norm.item(),
                "actor/q_action_grad_disagreement": q_action_grad_disagreement.item(),
                "actor/mean_loc": loc.abs().mean().item(),
                "actor/mean_scale": scale.mean().item(),
            }
            if self.rb_prior is not None:
                assert advantage is not None and horizon_mask is not None
                if self.cfg.bc_coef != 0.0:
                    assert bc_gate is not None
                    infos["rlpd/bc_term"] = bc_term[B_online: B_online + B_prior].mean().item()
                    infos["rlpd/bc_frac"] = bc_gate.float().mean().item()
                mask = horizon_mask.reshape(-1)
                if mask.any():
                    infos["rlpd/online_advantage"] = advantage[mask].mean().item()
                    if group_advantage is not None:
                        rows = group_advantage[mask]
                        for i, name in enumerate(self.reward_group_keys):
                            infos[f"rlpd/online_advantage/{name}"] = rows[:, i].mean().item()
                else:
                    infos["rlpd/online_advantage"] = float("nan")
                    if group_advantage is not None:
                        for name in self.reward_group_keys:
                            infos[f"rlpd/online_advantage/{name}"] = float("nan")

        return infos

    def _update_fb(self, batch: TensorDict) -> dict[str, float]:
        """One-step paired FB loss. The bootstrap action is the target actor.

        Gradients stay in ``F`` and ``B``. The actor loss never reads them.
        """
        obs = self.vecnorm_fb_obs(batch[OBS_KEY])
        next_obs = self.vecnorm_fb_obs(batch["next", OBS_KEY])
        next_state = self.vecnorm_b(batch["next", FB_KEY])
        action = batch[ACTION_KEY]
        z = batch["z"]
        terminated = batch["next", "terminated"].float()
        discount = batch["next", "discount"] * (1.0 - terminated) * self.cfg.gamma
        batch_size = next_state.shape[0]
        eye = torch.eye(batch_size, device=next_state.device)
        off_diag = 1.0 - eye
        off_diag_sum = off_diag.sum().clamp_min(1.0)

        with torch.no_grad():
            self.preproc(batch["next"])
            loc, scale = unwrap_ddp(self.actor_target)(batch["next", "_input_normed"])
            next_action = self.DistClass(loc, scale).sample()
            target_f = self.F_target(next_obs, next_action, z)
            target_b = self.B_target(next_state)
            target_m = uncertainty_target(
                torch.matmul(target_f, target_b.transpose(-1, -2)),
                self.cfg.fb_pessimism,
            )

        pred_f = self.F(obs, action, z)
        pred_b = self.B(next_state)
        pred_m = torch.matmul(pred_f, pred_b.transpose(-1, -2))
        diff = pred_m - discount.view(1, -1, 1) * target_m.unsqueeze(0)
        fb_offdiag = 0.5 * (diff * off_diag).square().sum() / off_diag_sum
        fb_diag = -torch.diagonal(diff, dim1=-2, dim2=-1).mean() * diff.shape[0]
        cov = torch.matmul(pred_b, pred_b.transpose(-1, -2))
        orth_diag = -cov.diagonal().mean()
        orth_off = 0.5 * (cov * off_diag).square().sum() / off_diag_sum
        orth = orth_off + orth_diag
        loss = fb_offdiag + fb_diag + self.cfg.ortho_coef * orth

        with torch.no_grad():
            next_q = uncertainty_target((target_f * z).sum(-1), self.cfg.fb_pessimism)
            feature_cov = torch.matmul(pred_b.transpose(-1, -2), pred_b) / batch_size
            solved = torch.linalg.solve(feature_cov.float(), pred_b.float(), left=False)
            reward = (solved * z.float()).sum(-1)
            q_target = reward + discount.squeeze(-1) * next_q
        q_pred = (pred_f * z).sum(-1)
        q_loss = 0.5 * pred_f.shape[0] * F.mse_loss(q_pred, q_target.unsqueeze(0).expand_as(q_pred))
        loss = loss + self.cfg.q_loss_coef * q_loss

        self.opt_F.zero_grad(set_to_none=True)
        self.opt_B.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm_F = nn.utils.clip_grad_norm_(self.F.parameters(), self.cfg.max_grad_norm)
        grad_norm_B = nn.utils.clip_grad_norm_(self.B.parameters(), self.cfg.max_grad_norm)
        self.opt_F.step()
        self.opt_B.step()
        with torch.no_grad():
            q_heads = (pred_f * z).sum(dim=-1)
            q_behavior = q_heads.mean(dim=0)
            q_disagreement = (q_heads - q_behavior.unsqueeze(0)).abs().mean()
            m_diag = torch.diagonal(pred_m, dim1=-2, dim2=-1).mean()
        return {
            "fb/loss": float(loss.detach()),
            "fb/offdiag": float(fb_offdiag.detach()),
            "fb/diag": float(fb_diag.detach()),
            "fb/orth_off": float(orth_off.detach()),
            "fb/q_value": float(q_behavior.mean()),
            "fb/q_std": float(q_behavior.std()),
            "fb/q_disagreement": float(q_disagreement),
            "fb/m_diag": float(m_diag),
            "fb/discount": float(discount.mean()),
            "fb/q_loss": float(q_loss.detach()),
            "fb/grad_norm_F": float(grad_norm_F),
            "fb/grad_norm_B": float(grad_norm_B),
        }

    def encode_reference(self, ref_state: torch.Tensor) -> torch.Tensor:
        """``z = Π(mean_h B(s_ref))`` over ``BFMCommand.ref_fb_state``."""
        if ref_state.shape[-1] != self.b_dim:
            raise ValueError(f"ref_fb_state dim {ref_state.shape[-1]} != fb_state group {self.b_dim}")
        flat = self.vecnorm_b(ref_state.reshape(-1, self.b_dim))
        encoded = unwrap_ddp(self.B)(flat).reshape(*ref_state.shape[:-1], self.z_dim)
        return project_z(encoded.mean(dim=1))

    def state_dict(self):
        state_dict = OrderedDict()
        # Save the underlying modules so checkpoints are portable between
        # distributed and single-process runs.
        state_dict["Q"] = unwrap_ddp(self.Q).state_dict()
        state_dict["actor"] = unwrap_ddp(self.actor).state_dict()
        state_dict["alpha"] = unwrap_ddp(self.alpha).state_dict()
        state_dict["vecnorm_obs"] = self.vecnorm_obs.state_dict()
        state_dict["F"] = unwrap_ddp(self.F).state_dict()
        state_dict["B"] = unwrap_ddp(self.B).state_dict()
        state_dict["vecnorm_fb_obs"] = self.vecnorm_fb_obs.state_dict()
        state_dict["vecnorm_b"] = self.vecnorm_b.state_dict()
        if self.reward_normalizer is not None:
            state_dict["reward_normalizer"] = self.reward_normalizer.state_dict()
        return state_dict

    def load_state_dict(self, state_dict: dict, strict: bool = True):
        unwrap_ddp(self.Q).load_state_dict(state_dict["Q"], strict=strict)
        unwrap_ddp(self.actor).load_state_dict(state_dict["actor"], strict=strict)
        if "opt_alpha" in state_dict:
            self.opt_alpha.load_state_dict(state_dict["opt_alpha"])
        alpha = unwrap_ddp(self.alpha)
        if "alpha" in state_dict:
            alpha.load_state_dict(state_dict["alpha"], strict=strict)
        elif "log_alpha" in state_dict:
            alpha.log_alpha.data = state_dict["log_alpha"].to(self.device)
        self.vecnorm_obs.load_state_dict(state_dict["vecnorm_obs"])
        unwrap_ddp(self.F).load_state_dict(state_dict["F"], strict=strict)
        unwrap_ddp(self.B).load_state_dict(state_dict["B"], strict=strict)
        self.vecnorm_fb_obs.load_state_dict(state_dict["vecnorm_fb_obs"])
        self.vecnorm_b.load_state_dict(state_dict["vecnorm_b"])
        self.F_target.load_state_dict(unwrap_ddp(self.F).state_dict())
        self.B_target.load_state_dict(unwrap_ddp(self.B).state_dict())
        rk = state_dict.get("reward_normalizer")
        if self.reward_normalizer is not None and rk is not None:
            self.reward_normalizer.load_state_dict(rk)


class SACFBRolloutPolicy(TensorDictModuleBase):
    """SAC rollout, plus the reference prompt stored for the FB update."""

    def __init__(self, policy: TensorDictModuleBase, algo: SACFB, eval_mode: bool):
        super().__init__()
        self.policy = policy
        self.algo = algo
        self.eval_mode = eval_mode
        self.in_keys = list(policy.in_keys)
        self.out_keys = list(policy.out_keys) + ["z"]

    def forward(self, tensordict: TensorDict) -> TensorDict:
        algo = self.algo
        if self.eval_mode:
            with VecNorm.freeze():
                self._write_z(algo, tensordict)
                return self.policy(tensordict)
        self._write_z(algo, tensordict)
        return self.policy(tensordict)

    @staticmethod
    def _write_z(algo: SACFB, tensordict: TensorDict) -> None:
        algo.vecnorm_fb_obs(tensordict[OBS_KEY])
        algo.vecnorm_b(tensordict[FB_KEY])
        tensordict["z"] = algo.encode_reference(algo.command_manager.ref_fb_state)

