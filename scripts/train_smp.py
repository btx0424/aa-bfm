"""Train an SMP diffusion prior on the cached LAFAN1 G1 motions.

The score model is the G1 reproduction in SUZ-tsinghua/smp, not MimicKit's
TinyMDM trainer. Both implement Mu et al., arXiv:2512.03028. Differences that
change the prior or the reward built from it:

Original (MimicKit ``TinyMDMModel``, ``tinymdm_multi_clip.yaml``)
    Humanoid discriminator observations at 30 Hz, not this 59-d G1 layout.
    Running mean/std normalizer, std clipped at 0.2.
    ``diffusers.DDPMScheduler`` with ``squaredcos_cap_v2``, T = 50.
    One ``(t, ε)`` draw per window. Training loss is L1 on ε. The paper writes
    an L2 loss.
    EMA (decay 0.995, every 10 steps, after 5000 updates).
    Optional class-conditional DiT. The SDS term runs a DDIM step, reconstructs
    x0, and converts that prediction back to ε before the squared error.
    The policy reward is additive:
    ``task_reward_weight * task + smp_reward_weight * r_smp``.

This script (SUZ)
    50 Hz windows of length 10, anchored on the last frame's yaw. Each frame is
    ``root_pos(3), tan-norm root_rot(6), joint_pos(29), ee_pos(15),
    root_lin_vel(3), root_ang_vel(3)``.
    Tan-norm stores the rotation matrix's x and z columns. That is not Zhou's
    first-two-columns 6D used by the tracking command.
    Features are mapped to [-1, 1] with per-dimension q01/q99.
    Cosine betas with s = 0.008 and T = 50, the same family as
    ``squaredcos_cap_v2``, implemented without ``diffusers``.
    L1 ε-prediction with 10 noise samples per window. EMA is off. No CFG.
    The guidance reward, not trained here, is the direct residual
    ``exp(-w_s / |K| * sum_{t in K} ||ε̂_t - ε_t||^2)`` at fixed timesteps.
    SUZ multiplies that by the task return instead of adding a weighted copy.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
_EE_BODY_NAMES = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)
FEATURE_DIM = 3 + 6 + len(_JOINT_NAMES) + 3 * len(_EE_BODY_NAMES) + 3 + 3


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def _quat_rotate_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    xyz = quat[..., 1:]
    t = xyz.cross(vec, dim=-1) * 2
    return vec - quat[..., :1] * t + xyz.cross(t, dim=-1)


def _yaw_quat(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    out = torch.zeros_like(quat)
    out[..., 0] = torch.cos(yaw / 2)
    out[..., 3] = torch.sin(yaw / 2)
    return out


def _matrix_from_quat(quat: torch.Tensor) -> torch.Tensor:
    r, i, j, k = quat.unbind(-1)
    two_s = 2.0 / quat.square().sum(-1)
    return torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        dim=-1,
    ).reshape(quat.shape[:-1] + (3, 3))


def _tan_norm(quat: torch.Tensor) -> torch.Tensor:
    """Rotation as ``[x-column, z-column]``, the SUZ tan-norm layout."""
    mat = _matrix_from_quat(quat)
    return torch.cat((mat[..., :, 0], mat[..., :, 2]), dim=-1)


def _windows_from_clip(
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    root_lin_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    joint_pos: torch.Tensor,
    ee_pos: torch.Tensor,
    window_size: int,
    stride: int,
) -> torch.Tensor | None:
    length = root_pos.shape[0]
    if length < window_size:
        return None
    starts = torch.arange(0, length - window_size + 1, stride)
    offsets = torch.arange(window_size)
    index = starts[:, None] + offsets[None, :]
    pos = root_pos[index]
    quat = root_quat[index]
    lin_vel = root_lin_vel[index]
    ang_vel = root_ang_vel[index]
    joints = joint_pos[index]
    ee = ee_pos[index]
    count, width, num_ee, _ = ee.shape

    anchor_pos = pos[:, -1]
    yaw = _yaw_quat(quat[:, -1])
    yaw_inv = _quat_conjugate(yaw)
    yaw_w = yaw[:, None].expand(count, width, 4).reshape(-1, 4)
    yaw_inv_w = yaw_inv[:, None].expand(count, width, 4).reshape(-1, 4)

    root_local = _quat_rotate_inverse(yaw_w, (pos - anchor_pos[:, None]).reshape(-1, 3))
    root_local = root_local.reshape(count, width, 3).clone()
    root_local[..., 2] = pos[..., 2]
    root_rot = _tan_norm(_quat_mul(yaw_inv_w, quat.reshape(-1, 4)).reshape(count, width, 4))

    ee_delta = ee - pos[:, :, None, :]
    yaw_ee = yaw[:, None, None].expand(count, width, num_ee, 4).reshape(-1, 4)
    ee_local = _quat_rotate_inverse(yaw_ee, ee_delta.reshape(-1, 3)).reshape(count, width, -1)
    lin_local = _quat_rotate_inverse(yaw_w, lin_vel.reshape(-1, 3)).reshape(count, width, 3)
    ang_local = _quat_rotate_inverse(yaw_w, ang_vel.reshape(-1, 3)).reshape(count, width, 3)
    return torch.cat((root_local, root_rot, joints, ee_local, lin_local, ang_local), dim=-1)


def load_windows(
    archive: Path,
    window_size: int,
    stride: int,
    clip_regex: str | None,
    max_clips: int | None,
) -> torch.Tensor:
    motions = np.load(archive, allow_pickle=False)
    body_names = [str(name) for name in motions["body_names"].tolist()]
    missing = [name for name in _EE_BODY_NAMES if name not in body_names]
    if missing:
        raise KeyError(f"Motion archive is missing end-effectors {missing}")
    ee_index = [body_names.index(name) for name in _EE_BODY_NAMES]
    joint_names = [str(name) for name in motions["joint_names"].tolist()]
    if tuple(joint_names) != _JOINT_NAMES:
        raise ValueError("Archive joint order does not match the G1 CSV order")

    clip_names = [str(name) for name in motions["clip_names"].tolist()]
    pattern = re.compile(clip_regex) if clip_regex else None
    keep = [
        index
        for index, name in enumerate(clip_names)
        if pattern is None or pattern.search(name)
    ]
    if max_clips is not None:
        keep = keep[:max_clips]
    if not keep:
        raise ValueError(f"No clips matched {clip_regex!r}")

    root_pos = torch.from_numpy(motions["root_pos"])
    root_quat = torch.from_numpy(motions["root_quat"])
    root_lin_vel = torch.from_numpy(motions["root_lin_vel"])
    root_ang_vel = torch.from_numpy(motions["root_ang_vel"])
    joint_pos = torch.from_numpy(motions["joint_pos"])
    body_pos = torch.from_numpy(motions["body_pos"])
    starts = motions["starts"]
    ends = motions["ends"]
    chunks = []
    for index in keep:
        frame = slice(int(starts[index]), int(ends[index]))
        windows = _windows_from_clip(
            root_pos[frame],
            root_quat[frame],
            root_lin_vel[frame],
            root_ang_vel[frame],
            joint_pos[frame],
            body_pos[frame][:, ee_index],
            window_size,
            stride,
        )
        if windows is not None:
            chunks.append(windows)
            print(f"{clip_names[index]}: {windows.shape[0]} windows", flush=True)
    if not chunks:
        raise ValueError("Every selected clip is shorter than the window")
    windows = torch.cat(chunks, dim=0)
    if windows.shape[-1] != FEATURE_DIM:
        raise RuntimeError(f"Expected feature dim {FEATURE_DIM}, got {windows.shape[-1]}")
    return windows


def quantile_normalize(windows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat = windows.reshape(-1, windows.shape[-1])
    q_low = torch.quantile(flat, 0.01, dim=0)
    q_high = torch.quantile(flat, 0.99, dim=0)
    tiny = (q_high - q_low) < 1.0e-6
    q_high = torch.where(tiny, q_low + 1.0, q_high)
    normalized = 2.0 * (windows - q_low) / (q_high - q_low) - 1.0
    return normalized, q_low, q_high


def _cosine_betas(num_timesteps: int, max_beta: float = 0.999) -> torch.Tensor:
    def alpha_bar(t: float) -> float:
        return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for index in range(num_timesteps):
        t1 = index / num_timesteps
        t2 = (index + 1) / num_timesteps
        betas.append(min(1.0 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


class DDPMScheduler(nn.Module):
    def __init__(self, num_timesteps: int = 50) -> None:
        super().__init__()
        self.num_timesteps = num_timesteps
        betas = _cosine_betas(num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        view = (-1, *([1] * (x0.ndim - 1)))
        return (
            self.sqrt_alphas_cumprod[t].view(view) * x0
            + self.sqrt_one_minus_alphas_cumprod[t].view(view) * noise
        )

    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(0, self.num_timesteps, (batch_size,), device=device)


class _Timesteps(nn.Module):
    def __init__(self, num_channels: int = 256) -> None:
        super().__init__()
        self.num_channels = num_channels

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.num_channels // 2
        exponent = -math.log(10000.0) * torch.arange(half, device=t.device) / half
        emb = t.float()[:, None] * torch.exp(exponent)[None, :]
        return torch.cat((torch.cos(emb), torch.sin(emb)), dim=-1)


class _DiTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        inner = dim * 4
        self.ff_in = nn.Linear(dim, inner * 2)
        self.ff_out = nn.Linear(inner, dim)
        self.ff_dropout = nn.Dropout(dropout)
        self.scale_shift_table = nn.Parameter(torch.randn(1, 1, 6, dim) / dim**0.5)

    def forward(self, x: torch.Tensor, time_hidden: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table + time_hidden.reshape(x.shape[0], 1, 6, -1)
        ).chunk(6, dim=-2)
        h = self.norm1(x) * (1 + scale_msa.squeeze(-2)) + shift_msa.squeeze(-2)
        b, n, _ = h.shape
        heads, head_dim = self.num_heads, self.head_dim
        q = self.to_q(h).reshape(b, n, heads, head_dim).transpose(1, 2)
        k = self.to_k(h).reshape(b, n, heads, head_dim).transpose(1, 2)
        v = self.to_v(h).reshape(b, n, heads, head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = self.attn_dropout(self.to_out(attn.transpose(1, 2).reshape(b, n, -1)))
        x = x + gate_msa.squeeze(-2) * attn
        h = self.norm2(x) * (1 + scale_mlp.squeeze(-2)) + shift_mlp.squeeze(-2)
        a, gate = self.ff_in(h).chunk(2, dim=-1)
        ff = self.ff_dropout(self.ff_out(F.silu(a) * gate))
        return x + gate_mlp.squeeze(-2) * ff


class DiffusionDenoiser(nn.Module):
    """ε-prediction DiT. Input and output are ``(batch, window, feature)``."""

    def __init__(
        self,
        feature_dim: int,
        window_size: int,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model {d_model} must be divisible by nhead {nhead}")
        self.preprocess = nn.Conv1d(feature_dim, feature_dim, 1, bias=False)
        self.proj_in = nn.Linear(feature_dim, d_model, bias=False)
        self.time_proj = _Timesteps(256)
        self.time_mlp = nn.Sequential(
            nn.Linear(256, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model),
        )
        position = torch.arange(max(window_size, 32)).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max(window_size, 32), d_model)
        pe[0, :, 0::2] = torch.sin(position * div)
        pe[0, :, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe, persistent=False)
        self.blocks = nn.ModuleList(_DiTBlock(d_model, nhead, dropout) for _ in range(num_layers))
        self.proj_out = nn.Linear(d_model, feature_dim, bias=False)
        self.postprocess = nn.Conv1d(feature_dim, feature_dim, 1, bias=False)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = x_t.transpose(1, 2)
        h = (self.preprocess(h) + h).transpose(1, 2)
        h = self.proj_in(h) + self.pe[:, : x_t.shape[1]]
        time_hidden = self.time_mlp(self.time_proj(t)).unsqueeze(1)
        for block in self.blocks:
            h = block(h, time_hidden)
        h = self.proj_out(h).transpose(1, 2)
        return (self.postprocess(h) + h).transpose(1, 2)


def diffusion_loss(
    model: DiffusionDenoiser,
    scheduler: DDPMScheduler,
    x0: torch.Tensor,
    num_noise_samples: int,
) -> torch.Tensor:
    batch = x0.shape[0]
    expanded = x0[:, None].expand(batch, num_noise_samples, *x0.shape[1:])
    expanded = expanded.reshape(batch * num_noise_samples, *x0.shape[1:])
    t = scheduler.sample_timesteps(expanded.shape[0], x0.device)
    noise = torch.randn_like(expanded)
    x_t = scheduler.add_noise(expanded, noise, t)
    return F.l1_loss(model(x_t, t), noise)


@torch.no_grad()
def score_residual(
    model: DiffusionDenoiser,
    scheduler: DDPMScheduler,
    x0: torch.Tensor,
    timesteps: tuple[int, ...],
) -> torch.Tensor:
    """Mean squared ε error at fixed timesteps. Shape ``(batch,)``.

    SUZ's guidance reward is ``exp(-w_s * this)``. MimicKit instead routes the
    prediction through one DDIM step before comparing ε.
    """
    total = torch.zeros(x0.shape[0], device=x0.device)
    for step in timesteps:
        t = torch.full((x0.shape[0],), step, dtype=torch.long, device=x0.device)
        noise = torch.randn_like(x0)
        eps_hat = model(scheduler.add_noise(x0, noise, t), t)
        total = total + (eps_hat - noise).square().mean(dim=(-1, -2))
    return total / len(timesteps)


def train(args: argparse.Namespace) -> Path:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    windows = load_windows(
        Path(args.archive),
        args.window_size,
        args.stride,
        args.clip_regex,
        args.max_clips,
    )
    windows, q_low, q_high = quantile_normalize(windows)
    print(f"windows {tuple(windows.shape)}", flush=True)
    model = DiffusionDenoiser(
        FEATURE_DIM,
        args.window_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    scheduler = DDPMScheduler(args.num_timesteps).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    count = windows.shape[0]
    for epoch in range(args.epochs):
        order = torch.randperm(count)
        total = 0.0
        batches = 0
        for start in range(0, count, args.batch_size):
            batch = windows[order[start : start + args.batch_size]].to(device)
            loss = diffusion_loss(model, scheduler, batch, args.num_noise_samples)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        print(f"epoch {epoch:4d}  loss {total / batches:.6f}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "q_low": q_low,
            "q_high": q_high,
            "window_size": args.window_size,
            "feature_dim": FEATURE_DIM,
            "ee_body_names": _EE_BODY_NAMES,
            "num_timesteps": args.num_timesteps,
            "d_model": args.d_model,
            "nhead": args.nhead,
            "num_layers": args.num_layers,
        },
        out,
    )
    print(f"saved {out}", flush=True)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=_repo_root() / "data" / "lafan1" / "motions.npz",
    )
    parser.add_argument("--out", type=Path, default=_repo_root() / "outputs" / "smp" / "prior.pt")
    parser.add_argument("--clip-regex", default=None)
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--num-timesteps", type=int, default=50)
    parser.add_argument("--num-noise-samples", type=int, default=10)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
