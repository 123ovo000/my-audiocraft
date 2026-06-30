import os
import xformers  # 加速
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from audiocraft.models import MusicGen
from audiocraft.models.loaders import load_compression_model, load_lm_model
from audiocraft.modules.conditioners import ConditioningAttributes
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# --- 本地 G 盘路径配置 ---
MODEL_PATH = r"G:\audiocraft-main\models\musicgen-small"
CACHE_DIR = r"G:\audiocraft-main\cache"
OUTPUT_DIR = r"G:\audiocraft-main\outputs\my_finetune"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GRAD_ACC = 4
NUM_EPOCHS = 50
MAX_LR = 1e-4


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


class SimplePtDataset(Dataset):
    def __init__(self, cache_dir):
        self.file_paths = torch.load(os.path.join(cache_dir, "index.pt"))

    def __len__(self): return len(self.file_paths)

    def __getitem__(self, idx): return torch.load(self.file_paths[idx])


def main():
    set_seed(42)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    writer = SummaryWriter(os.path.join(OUTPUT_DIR, "logs"))

    print(f"Step 1: 在 {DEVICE} 上加载模型 (已启用 xformers 优化)...")
    compression_model = load_compression_model(MODEL_PATH, device=DEVICE)
    lm = load_lm_model(MODEL_PATH, device=DEVICE)

    lm.float();
    compression_model.float()
    model = MusicGen(name="musicgen-small", lm=lm, compression_model=compression_model)

    model.compression_model.eval()
    model.lm.train()
    for p in model.compression_model.parameters(): p.requires_grad = False

    # 划分训练/验证集
    full_dataset = SimplePtDataset(CACHE_DIR)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    # 🔥 关键修改：collate_fn 改为 x[0]，直接返回字典，不再是列表
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=lambda x: x[0])
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])

    optimizer = torch.optim.AdamW(model.lm.parameters(), lr=MAX_LR)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=MAX_LR, total_steps=NUM_EPOCHS * len(train_loader) // GRAD_ACC, pct_start=0.1
    )

    print("Step 2: 开始训练...")
    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        model.lm.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch + 1}")

        for i, data in enumerate(pbar):
            # 🔥 这里的 data 现在一定是字典，不再需要 isinstance 判断
            tokens = data['tokens'].to(DEVICE).long()
            desc = str(data['description'])

            attr = ConditioningAttributes(text={'description': desc})
            logits = model.lm(tokens, [attr])

            loss = F.cross_entropy(
                logits.permute(0, 1, 3, 2).float().reshape(-1, logits.shape[-1]),
                tokens.reshape(-1)
            ) / GRAD_ACC

            loss.backward()

            if (i + 1) % GRAD_ACC == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()

            train_loss += loss.item() * GRAD_ACC
            pbar.set_postfix({"Loss": f"{loss.item() * GRAD_ACC:.4f}"})

        # 验证过程
        model.lm.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                # 🔥 验证集现在也是字典
                tokens = data['tokens'].to(DEVICE).long()
                attr = ConditioningAttributes(text={'description': str(data['description'])})
                logits = model.lm(tokens, [attr])
                loss = F.cross_entropy(logits.permute(0, 1, 3, 2).float().reshape(-1, logits.shape[-1]),
                                       tokens.reshape(-1))
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        writer.add_scalar("Loss/Train", train_loss / len(train_loader), epoch)
        writer.add_scalar("Loss/Val", avg_val_loss, epoch)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.lm.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pt"))
            print(f" -> 发现更优模型，已保存！Val Loss: {avg_val_loss:.4f}")


if __name__ == "__main__":
    main()