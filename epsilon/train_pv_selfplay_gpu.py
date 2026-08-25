import argparse
from datetime import datetime
import os

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from torch.amp import autocast, GradScaler
import torch.backends.cudnn as cudnn

from pv_model import LinithPVNet
from action_space import ACTION_SIZE
from action_space import (
    MAX_SWANS,
    SUBSET_MASKS,
    MASK_TO_INDEX,
    PLACE_SWAN_START,
    PLACE_SWAN_END,
    PLACE_STONE_START,
    PLACE_STONE_END,
    MOVE_GROUP_START,
    MOVE_GROUP_END,
    PUSH_START,
    PUSH_END,
)
from linithrules import DIRS8, BOARD_SIZE


# ---------------------------------------------------------
# Atreyan era helper (for logging symmetry with hyperloop style)
# ---------------------------------------------------------
def format_dt_ae(dt: datetime) -> str:
    ae = dt.year - 2020
    return f"{ae}AE-{dt:%m-%d %H:%M:%S}"


# ---------------------------------------------------------
# Symmetry helpers (precomputed once)
# ---------------------------------------------------------
def _build_symmetry_maps():
    # operate on last 2 dims (H,W)
    def t_id(a):        return a
    def t_rot90(a):     return np.rot90(a, k=1, axes=(-2, -1))
    def t_rot180(a):    return np.rot90(a, k=2, axes=(-2, -1))
    def t_rot270(a):    return np.rot90(a, k=3, axes=(-2, -1))
    def t_flip_h(a):    return np.flip(a, axis=-1)
    def t_flip_v(a):    return np.flip(a, axis=-2)
    def t_transpose(a): return np.swapaxes(a, -1, -2)
    def t_anti(a):      return np.rot90(np.swapaxes(a, -1, -2), k=2, axes=(-2, -1))

    sym_defs = [
        ("id",        t_id),
        ("rot90",     t_rot90),
        ("rot180",    t_rot180),
        ("rot270",    t_rot270),
        ("flip_h",    t_flip_h),
        ("flip_v",    t_flip_v),
        ("transpose", t_transpose),
        ("anti",      t_anti),
    ]

    # track where each original square (0..99) goes
    base_squares = np.arange(BOARD_SIZE * BOARD_SIZE, dtype=np.int32).reshape(
        1, BOARD_SIZE, BOARD_SIZE
    )
    dirs = DIRS8
    sym_info = []

    for name, tf in sym_defs:
        transformed = tf(base_squares)
        flat = transformed.reshape(-1)

        # square_map[old_square] = new_square
        square_map = np.empty(BOARD_SIZE * BOARD_SIZE, dtype=np.int32)
        for new_index in range(BOARD_SIZE * BOARD_SIZE):
            old_val = flat[new_index]
            square_map[old_val] = new_index

        # direction mapping: transform the 8 neighbours of the center
        center_r = BOARD_SIZE // 2
        center_c = BOARD_SIZE // 2
        center_idx = center_r * BOARD_SIZE + center_c
        center_new = square_map[center_idx]
        center_new_r, center_new_c = divmod(center_new, BOARD_SIZE)

        dir_map = [0] * len(dirs)
        for d_idx, (dr, dc) in enumerate(dirs):
            r1 = center_r + dr
            c1 = center_c + dc
            assert 0 <= r1 < BOARD_SIZE and 0 <= c1 < BOARD_SIZE
            s1 = r1 * BOARD_SIZE + c1
            s1_new = square_map[s1]
            r1_new, c1_new = divmod(s1_new, BOARD_SIZE)
            dr_new, dc_new = r1_new - center_new_r, c1_new - center_new_c

            # find which DIRS8 this turned into
            found = None
            for j, (dr2, dc2) in enumerate(dirs):
                if dr2 == dr_new and dc2 == dc_new:
                    found = j
                    break
            if found is None:
                raise RuntimeError(f"Could not map direction {dr,dc} for symmetry {name}")
            dir_map[d_idx] = found

        sym_info.append(
            {
                "name": name,
                "tf": tf,
                "square_map": square_map,
                "dir_map": np.array(dir_map, dtype=np.int32),
            }
        )

    return sym_info


_SYM_INFO = None


def get_sym_info():
    global _SYM_INFO
    if _SYM_INFO is None:
        _SYM_INFO = _build_symmetry_maps()
    return _SYM_INFO


def _compute_swan_permutation(x_single, square_map):
    """
    x_single : (6,10,10) for one position

    Returns perm[i] = new index of old swan i in the sorted-by-square order.
    """
    # channel 5 is 'current player' (1.0 = SUN, 0.0 = MOON)
    cur_is_sun = x_single[5, 0, 0] > 0.5
    active_chan = 0 if cur_is_sun else 2  # SUN active / MOON active

    swan_squares = []
    for r in range(BOARD_SIZE):
        row = x_single[active_chan, r, :]
        cols = np.nonzero(row > 0.5)[0]
        for c in cols:
            swan_squares.append(r * BOARD_SIZE + c)

    swan_squares = sorted(swan_squares)
    n = len(swan_squares)
    if n == 0:
        # no swans -> group/push actions are illegal anyway
        return np.arange(MAX_SWANS, dtype=np.int32)

    if n > MAX_SWANS:
        swan_squares = swan_squares[:MAX_SWANS]
        n = MAX_SWANS

    # map each swan square through symmetry, then re-sort
    new_squares_unsorted = [int(square_map[s]) for s in swan_squares]
    new_squares_sorted = sorted(new_squares_unsorted)
    new_index_from_square = {s: idx for idx, s in enumerate(new_squares_sorted)}

    perm = np.arange(MAX_SWANS, dtype=np.int32)
    for i, s_old in enumerate(swan_squares):
        s_new = new_squares_unsorted[i]
        j = new_index_from_square[s_new]
        perm[i] = j
    return perm


def augment_sample(x, pi, z, sym_index=None):
    """
    Apply one random (or chosen) D4 symmetry to a *single* sample.

    x  : (6,10,10)
    pi : (ACTION_SIZE,)
    z  : scalar
    """
    x = np.asarray(x, dtype=np.float32)
    pi = np.asarray(pi, dtype=np.float32)

    sym_info = get_sym_info()
    if sym_index is None:
        sym_index = np.random.randint(len(sym_info))

    sym = sym_info[sym_index]
    tf = sym["tf"]
    square_map = sym["square_map"]
    dir_map = sym["dir_map"]
    dirs = DIRS8

    # transform board
    x_sym = tf(x)

    # swan index permutation
    perm = _compute_swan_permutation(x, square_map)

    # remap policy
    pi_sym = np.zeros_like(pi)

    # --- 1) placement actions ---
    squares = np.arange(BOARD_SIZE * BOARD_SIZE, dtype=np.int32)
    mapped_squares = square_map[squares]

    # place swan (0..99)
    pi_sym[PLACE_SWAN_START + mapped_squares] = pi[PLACE_SWAN_START + squares]

    # place stone (100..199)
    pi_sym[PLACE_STONE_START + mapped_squares] = pi[PLACE_STONE_START + squares]

    # --- 2) group moves (subset mask + direction) ---
    num_subset = len(SUBSET_MASKS)
    for subset_idx in range(num_subset):
        mask = SUBSET_MASKS[subset_idx]

        # permute bitmask via Swan permutation
        new_mask = 0
        for bit in range(MAX_SWANS):
            if mask & (1 << bit):
                new_mask |= (1 << int(perm[bit]))

        new_subset_idx = MASK_TO_INDEX[new_mask]

        for dir_idx in range(len(dirs)):
            src_idx = MOVE_GROUP_START + subset_idx * len(dirs) + dir_idx
            v = pi[src_idx]
            if v == 0.0:
                continue
            new_dir_idx = int(dir_map[dir_idx])
            dst_idx = MOVE_GROUP_START + new_subset_idx * len(dirs) + new_dir_idx
            pi_sym[dst_idx] += v

    # --- 3) pushes (one swan index + direction) ---
    for swan_idx in range(MAX_SWANS):
        new_swan_idx = int(perm[swan_idx])
        for dir_idx in range(len(dirs)):
            src_idx = PUSH_START + swan_idx * len(dirs) + dir_idx
            v = pi[src_idx]
            if v == 0.0:
                continue
            new_dir_idx = int(dir_map[dir_idx])
            dst_idx = PUSH_START + new_swan_idx * len(dirs) + new_dir_idx
            pi_sym[dst_idx] += v

    x_sym = np.ascontiguousarray(x_sym, dtype=np.float32)
    pi_sym = np.ascontiguousarray(pi_sym, dtype=np.float32)

    return x_sym, pi_sym, float(z)


# ---------------------------------------------------------
# Dataset (uses optional symmetry)
# ---------------------------------------------------------
class SelfPlayDataset(Dataset):
    def __init__(self, X, Pi, Z, use_symmetry: bool = False):
        # np.array(..., copy=False) lets us keep memmap without forcing a full copy
        self.X = np.array(X, dtype=np.float32, copy=False)
        self.Pi = np.array(Pi, dtype=np.float32, copy=False)
        self.Z = np.array(Z, dtype=np.float32, copy=False)

        assert self.X.shape[0] == self.Pi.shape[0] == self.Z.shape[0]
        assert self.X.shape[1:] == (6, 10, 10)
        assert self.Pi.shape[1] == ACTION_SIZE

        self.use_symmetry = use_symmetry

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]   # (6,10,10)
        pi = self.Pi[idx] # (A,)
        z = self.Z[idx]   # ()

        if self.use_symmetry:
            x, pi, z = augment_sample(x, pi, z)

        return (
            torch.from_numpy(x),
            torch.from_numpy(pi),
            torch.tensor(z, dtype=torch.float32),
        )

# ---------------------------------------------------------
# Symmetry augmentation
# ---------------------------------------------------------
def apply_symmetry_augmentation(X, Pi, Z):
    """
    Real 8-way board symmetry augmentation for (X, Pi, Z).

    X  : (N, 6, 10, 10) float32 state tensors
    Pi : (N, ACTION_SIZE) policy over encoded actions
    Z  : (N,) value from current player's perspective

    For each of the 8 elements of D4 (rotations + flips),
    we transform both the board and the policy so that the
    augmented samples are consistent with the C++ action
    encoding (subset-based group moves + pushes).
    """
    X = np.asarray(X, dtype=np.float32)
    Pi = np.asarray(Pi, dtype=np.float32)
    Z = np.asarray(Z, dtype=np.float32)

    N, C, H, W = X.shape
    assert C == 6 and H == BOARD_SIZE and W == BOARD_SIZE
    assert Pi.shape[1] == ACTION_SIZE

    # Import action-space details so we don't duplicate constants
    from action_space import (
        MAX_SWANS,
        SUBSET_MASKS,
        MASK_TO_INDEX,
        PLACE_SWAN_START,
        PLACE_SWAN_END,
        PLACE_STONE_START,
        PLACE_STONE_END,
        MOVE_GROUP_START,
        MOVE_GROUP_END,
        PUSH_START,
        PUSH_END,
    )

    # ---------- 1. Define raw spatial transforms (operate on last 2 dims) ----------
    def t_id(a):
        return a

    def t_rot90(a):
        return np.rot90(a, k=1, axes=(-2, -1))

    def t_rot180(a):
        return np.rot90(a, k=2, axes=(-2, -1))

    def t_rot270(a):
        return np.rot90(a, k=3, axes=(-2, -1))

    def t_flip_h(a):
        # mirror left-right (c -> W-1-c)
        return np.flip(a, axis=-1)

    def t_flip_v(a):
        # mirror top-bottom (r -> H-1-r)
        return np.flip(a, axis=-2)

    def t_transpose(a):
        # main diagonal (r,c) -> (c,r)
        return np.swapaxes(a, -1, -2)

    def t_anti(a):
        # anti-diagonal (r,c) -> (H-1-c, W-1-r)
        return np.rot90(np.swapaxes(a, -1, -2), k=2, axes=(-2, -1))

    sym_defs = [
        ("id",        t_id),
        ("rot90",     t_rot90),
        ("rot180",    t_rot180),
        ("rot270",    t_rot270),
        ("flip_h",    t_flip_h),
        ("flip_v",    t_flip_v),
        ("transpose", t_transpose),
        ("anti",      t_anti),
    ]

    # ---------- 2. Precompute square and direction maps for each sym ----------
    # Square indices 0..99 with value = index so we can track where each square goes.
    base_squares = np.arange(BOARD_SIZE * BOARD_SIZE, dtype=np.int32).reshape(
        1, BOARD_SIZE, BOARD_SIZE
    )

    # DIRS8 as list of (dr,dc)
    dirs = DIRS8
    assert len(dirs) == 8

    sym_info = []
    for name, tf in sym_defs:
        # Square mapping: "old square index -> new square index"
        transformed = tf(base_squares)
        flat = transformed.reshape(-1)
        square_map = np.empty(BOARD_SIZE * BOARD_SIZE, dtype=np.int32)
        for new_index in range(BOARD_SIZE * BOARD_SIZE):
            old_val = flat[new_index]          # which original square ended up here
            square_map[old_val] = new_index    # position of that original square

        # Direction mapping (using an interior reference square)
        center_r = BOARD_SIZE // 2
        center_c = BOARD_SIZE // 2
        center_idx = center_r * BOARD_SIZE + center_c
        center_new = square_map[center_idx]
        center_new_r, center_new_c = divmod(center_new, BOARD_SIZE)

        dir_map = [0] * len(dirs)
        for d_idx, (dr, dc) in enumerate(dirs):
            r1 = center_r + dr
            c1 = center_c + dc
            # All 8 neighbours are in bounds for 10x10 with center at 5,5
            assert 0 <= r1 < BOARD_SIZE and 0 <= c1 < BOARD_SIZE
            s1 = r1 * BOARD_SIZE + c1
            s1_new = square_map[s1]
            r1_new, c1_new = divmod(s1_new, BOARD_SIZE)
            dr_new, dc_new = r1_new - center_new_r, c1_new - center_new_c
            # Find which DIRS8 this became
            found = None
            for j, (dr2, dc2) in enumerate(dirs):
                if dr2 == dr_new and dc2 == dc_new:
                    found = j
                    break
            if found is None:
                raise RuntimeError(f"Could not map direction {dr,dc} for symmetry {name}")
            dir_map[d_idx] = found

        sym_info.append(
            {
                "name": name,
                "tf": tf,
                "square_map": square_map,
                "dir_map": np.array(dir_map, dtype=np.int32),
            }
        )

    # ---------- 3. Helper: build swan index permutation for one position ----------
    def compute_swan_permutation(x_single, square_map):
        """
        x_single : (6,10,10) for one position
        square_map : length 100 array mapping old square -> new square

        Returns:
            perm : length MAX_SWANS array; perm[i] = new index of old swan i
        """
        # Current player channel is uniform 1.0 (SUN) or 0.0 (MOON)
        cur_is_sun = x_single[5, 0, 0] > 0.5
        active_chan = 0 if cur_is_sun else 2  # SUN active or MOON active

        # Collect all active swan squares for the current player
        swan_squares = []
        for r in range(BOARD_SIZE):
            row = x_single[active_chan, r, :]
            # In our encoding these are 0/1, but use >0.5 just in case
            cols = np.nonzero(row > 0.5)[0]
            for c in cols:
                swan_squares.append(r * BOARD_SIZE + c)

        swan_squares = sorted(swan_squares)
        n = len(swan_squares)
        if n == 0:
            # No active swans: group / push actions are all illegal anyway
            return np.arange(MAX_SWANS, dtype=np.int32)

        if n > MAX_SWANS:
            # Safety: mimic action encoder by only tracking first MAX_SWANS
            swan_squares = swan_squares[:MAX_SWANS]
            n = MAX_SWANS

        # Map each swan's square through the symmetry
        new_squares_unsorted = [int(square_map[s]) for s in swan_squares]
        new_squares_sorted = sorted(new_squares_unsorted)

        # Build lookup from new square -> its index in sorted list
        new_index_from_square = {
            s: idx for idx, s in enumerate(new_squares_sorted)
        }

        perm = np.arange(MAX_SWANS, dtype=np.int32)
        for i, s_old in enumerate(swan_squares):
            s_new = new_squares_unsorted[i]
            j = new_index_from_square[s_new]
            perm[i] = j

        return perm

    # ---------- 4. Actually augment the dataset ----------
    X_list = []
    Pi_list = []
    Z_list = []

    for idx in range(N):
        x = X[idx]          # (6,10,10)
        pi = Pi[idx]        # (ACTION_SIZE,)
        z = Z[idx]

        for sym in sym_info:
            tf = sym["tf"]
            square_map = sym["square_map"]
            dir_map = sym["dir_map"]

            # 4.1 Transform the board tensor
            x_sym = tf(x)

            # 4.2 Swan index permutation for this position / symmetry
            perm = compute_swan_permutation(x, square_map)

            # 4.3 Map the policy
            pi_sym = np.zeros_like(pi)

            # --- 4.3.1 Placement actions ---
            squares = np.arange(BOARD_SIZE * BOARD_SIZE, dtype=np.int32)

            # place_swan: 0..99
            mapped_squares = square_map[squares]
            pi_sym[PLACE_SWAN_START + mapped_squares] = pi[
                PLACE_SWAN_START + squares
            ]

            # place_stone: 100..199
            pi_sym[PLACE_STONE_START + mapped_squares] = pi[
                PLACE_STONE_START + squares
            ]

            # --- 4.3.2 Group moves (subset of up to 6 swans in DIRS8) ---
            num_subset = len(SUBSET_MASKS)
            for subset_idx in range(num_subset):
                mask = SUBSET_MASKS[subset_idx]

                # Remap the bitmask via the Swan permutation
                new_mask = 0
                for bit in range(MAX_SWANS):
                    if mask & (1 << bit):
                        new_mask |= (1 << int(perm[bit]))

                new_subset_idx = MASK_TO_INDEX[new_mask]

                for dir_idx in range(len(dirs)):
                    src_idx = MOVE_GROUP_START + subset_idx * len(dirs) + dir_idx
                    v = pi[src_idx]
                    if v == 0.0:
                        continue
                    new_dir_idx = int(dir_map[dir_idx])
                    dst_idx = (
                        MOVE_GROUP_START + new_subset_idx * len(dirs) + new_dir_idx
                    )
                    pi_sym[dst_idx] += v

            # --- 4.3.3 Push actions (one swan index + direction) ---
            for swan_idx in range(MAX_SWANS):
                new_swan_idx = int(perm[swan_idx])
                for dir_idx in range(len(dirs)):
                    src_idx = PUSH_START + swan_idx * len(dirs) + dir_idx
                    v = pi[src_idx]
                    if v == 0.0:
                        continue
                    new_dir_idx = int(dir_map[dir_idx])
                    dst_idx = (
                        PUSH_START + new_swan_idx * len(dirs) + new_dir_idx
                    )
                    pi_sym[dst_idx] += v

            X_list.append(x_sym)
            Pi_list.append(pi_sym)
            Z_list.append(z)

    X_aug = np.stack(X_list, axis=0)
    Pi_aug = np.stack(Pi_list, axis=0)
    Z_aug = np.stack(Z_list, axis=0)

    return X_aug, Pi_aug, Z_aug


# ---------------------------------------------------------
# Training loop
# ---------------------------------------------------------
def train_one_model(

    base_model_path: str,
    data_npz_path: str,
    out_model_path: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    lr_schedule: str,
    use_symmetry: bool,
    device: str = "cpu",
):
    dev = torch.device(device)
    use_amp = (dev.type == "cuda")

    if use_amp:
        # Let cuDNN pick fastest algorithms for this size
        cudnn.benchmark = True
        # Optional: make matmul kernels prefer TF32 / fast paths if available
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    print()
    print("========================")
    print("   Training New Model  ")
    print("========================")
    print()

    print(f"Loading data from {data_npz_path}")
    d = np.load(data_npz_path, mmap_mode="r")
    X = d["X"]
    Pi = d["Pi"]
    Z = d["Z"]

    print(f"Raw dataset shapes - X = {X.shape}, Pi = {Pi.shape}, Z = {Z.shape}")

    if use_symmetry:
        print("Symmetry augmentation ON (applied per-sample in the Dataset).")

    dataset = SelfPlayDataset(X, Pi, Z, use_symmetry=use_symmetry)

    # Enable pinned memory + a couple of workers when on CUDA
    pin_mem = (dev.type == "cuda")
    num_workers = 0

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=pin_mem,
        num_workers=num_workers,
    )

    # Load base model
    print(f"Loading base model from {base_model_path}")
    net = LinithPVNet()
    net.load_state_dict(torch.load(base_model_path, map_location=dev))
    net.to(dev)

    # Optimizer with L2 weight decay
    optimizer = optim.Adam(
        net.parameters(),
        lr=lr,
        weight_decay=weight_decay,  # L2
    )

    # AMP scaler for CUDA; disabled on CPU
    scaler = GradScaler(device="cuda") if use_amp else None

    # LR schedule
    scheduler = None
    lr_schedule = (lr_schedule or "none").lower()
    if lr_schedule == "step":
        # simple step schedule: decay by 0.5 every half of total epochs
        step_size = max(1, epochs // 2)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)
        print(f"Using StepLR schedule: step_size={step_size}, gamma=0.5")
    elif lr_schedule == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs)
        )
        print("Using CosineAnnealingLR schedule")
    else:
        print("Using constant learning rate (no LR schedule)")

    # Loss functions
    mse_loss = nn.MSELoss()

    start_time = datetime.now()
    print(f"Training start - {format_dt_ae(start_time)}")

    net.train()
    for epoch in range(1, epochs + 1):
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        epoch_total_loss = 0.0
        n_batches = 0

        for batch in loader:
            x, target_pi, target_z = batch

            # Use non_blocking transfers with pinned memory on CUDA
            x = x.to(dev, non_blocking=True)                 # [B,6,10,10]
            target_pi = target_pi.to(dev, non_blocking=True) # [B,A]
            target_z = target_z.to(dev, non_blocking=True)   # [B]

            optimizer.zero_grad(set_to_none=True)

            if use_amp and scaler is not None:
                # ----- AMP path (CUDA) -----
                with autocast("cuda", enabled=use_amp):
                    logits, value = net(x)            # logits: [B,A], value: [B,1]
                    value = value.view(-1)            # [B]

                    # policy loss: cross-entropy with soft targets Pi
                    log_probs = torch.log_softmax(logits, dim=1)
                    policy_loss = -(target_pi * log_probs).sum(dim=1).mean()

                    # value loss: MSE
                    value_loss = mse_loss(value, target_z)

                    loss = policy_loss + value_loss

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # ----- Regular FP32 path (CPU or if AMP disabled) -----
                logits, value = net(x)               # logits: [B,A], value: [B,1]
                value = value.view(-1)               # [B]

                log_probs = torch.log_softmax(logits, dim=1)
                policy_loss = -(target_pi * log_probs).sum(dim=1).mean()

                value_loss = mse_loss(value, target_z)
                loss = policy_loss + value_loss

                loss.backward()
                optimizer.step()

            epoch_policy_loss += policy_loss.item()
            epoch_value_loss += value_loss.item()
            epoch_total_loss += loss.item()
            n_batches += 1

        if n_batches > 0:
            epoch_policy_loss /= n_batches
            epoch_value_loss /= n_batches
            epoch_total_loss /= n_batches

        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch}/{epochs} "
            f"LR - {current_lr:.6f}  "
            f"Policy - {epoch_policy_loss:.4f}  "
            f"Value - {epoch_value_loss:.4f}  "
            f"Total - {epoch_total_loss:.4f}"
        )

    end_time = datetime.now()
    print(f"Training end   - {format_dt_ae(end_time)}")
    print(f"Duration       - {end_time - start_time}")

    # Save model
    out_dir = os.path.dirname(out_model_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    torch.save(net.state_dict(), out_model_path)
    print(f"Saved trained model to {out_model_path}")


# ---------------------------------------------------------
# Main / CLI
# ---------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Train Linith PV net from self-play data.")

    ap.add_argument("--base", type=str, required=True,
                    help="Base model .pt to start from.")
    ap.add_argument("--data", type=str, required=True,
                    help="Dataset .npz (X, Pi, Z). Can be a replay buffer file.")
    ap.add_argument("--out", type=str, required=True,
                    help="Output model .pt path.")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", type=str, default="cpu")

    # New knobs for hyperloop
    ap.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="L2 weight decay lambda.",
    )
    ap.add_argument(
        "--lr-schedule",
        type=str,
        default="none",
        help="LR schedule: 'none', 'step', or 'cosine'.",
    )
    ap.add_argument(
        "--symmetry-aug",
        action="store_true",
        help="Enable symmetry augmentation (rot/flip).",
    )

    args = ap.parse_args()

    train_one_model(
        base_model_path=args.base,
        data_npz_path=args.data,
        out_model_path=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lr_schedule=args.lr_schedule,
        use_symmetry=args.symmetry_aug,
        device=args.device,
    )


if __name__ == "__main__":
    main()
