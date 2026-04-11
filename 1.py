# train_lora.py - LoRA 微调 MusicGen
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_read
from audiocraft.utils.notebook import display_audio
import torch
import os

# 1. 准备你的音乐数据集
# 把你喜欢的音乐放在 data/your_music/ 目录下
music_dir = "data/your_music/"
music_files = [os.path.join(music_dir, f) for f in os.listdir(music_dir) if f.endswith('.wav')]

# 2. 加载预训练模型
model = MusicGen.get_pretrained('facebook/musicgen-small')
model.set_generation_params(duration=8)  # 生成8秒音乐

# 3. 准备训练数据
from audiocraft.data.audio_dataset import AudioDataset

dataset = AudioDataset(
    audio_files=music_files,
    sample_rate=32000,
    segment_duration=8  # 8秒片段
)

# 4. 使用 LoRA 微调（节省显存）
from audiocraft.utils.lora import apply_lora

apply_lora(model, rank=8)  # LoRA rank=8

# 5. 训练
from torch.utils.data import DataLoader

dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
model.train()

for epoch in range(10):  # 训练10个epoch
    for batch in dataloader:
        audio = batch['audio'].to('cuda' if torch.cuda.is_available() else 'cpu')

        # 前向传播
        loss = model(audio)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/10, Loss: {loss.item():.4f}")

# 6. 保存微调后的模型
torch.save(model.state_dict(), 'musicgen_finetuned.pth')