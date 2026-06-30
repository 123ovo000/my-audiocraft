import os
import torch
import torch.nn.functional as F
import json
import torchaudio
import random
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_read
from tqdm import tqdm

# --- 本地路径 ---
MODEL_PATH = r"G:\audiocraft-main\models\musicgen-small"
DATA_MANIFEST = r"G:\music_train_data\data\data.jsonl"
CACHE_DIR = r"G:\audiocraft-main\cache"


def precompute():
    with open(DATA_MANIFEST, 'r', encoding='utf-8') as f:
        data_list = [json.loads(line) for line in f]

    # 初始化模型
    print("加载模型中...")
    model = MusicGen.get_pretrained(MODEL_PATH, device="cuda")
    model.compression_model.eval()

    os.makedirs(CACHE_DIR, exist_ok=True)
    index_file = []

    segment_dur = 30.0

    print(f"开始处理 {len(data_list)} 首歌...")

    for item in tqdm(data_list, desc="预处理进度"):
        path = item['path']
        start_offset = item['start_time']  # 裁断前3s
        desc = item['description']

        try:
            metadata = torchaudio.info(path)
            file_duration = metadata.num_frames / metadata.sample_rate
        except Exception:
            continue

        effective_duration = file_duration - start_offset
        song_name = os.path.splitext(os.path.basename(path))[0]

        if effective_duration <= 0: continue

        # 逻辑：不足 30s 补0
        if effective_duration < segment_dur:
            audio, sr = audio_read(path, seek_time=start_offset, duration=effective_duration)
            target_frames = int(model.sample_rate * segment_dur)
            audio = F.pad(audio, (0, target_frames - audio.shape[-1]))

            with torch.no_grad():
                tokens, _ = model.compression_model.encode(audio.unsqueeze(0).to(model.device))

            save_path = os.path.join(CACHE_DIR, f"{song_name}_pad.pt")
            torch.save({'tokens': tokens.cpu(), 'description': desc}, save_path)
            index_file.append(save_path)

        # 逻辑：超过 30s，随机采样 2 段 (至少间隔 5s)
        else:
            # 采样空间：[start_offset, file_duration - 30]
            max_start = file_duration - segment_dur

            # 尝试随机生成两个满足条件的起始点
            # 逻辑：采样 S1, S2，保证 abs(S1 - S2) >= 5
            s1 = random.uniform(start_offset, max_start)

            # 尝试 100 次找到满足条件的 s2
            s2 = s1
            for _ in range(100):
                temp_s2 = random.uniform(start_offset, max_start)
                if abs(temp_s2 - s1) >= 5.0:
                    s2 = temp_s2
                    break

            for idx, start_pos in enumerate([s1, s2]):
                audio, sr = audio_read(path, seek_time=start_pos, duration=segment_dur)

                with torch.no_grad():
                    tokens, _ = model.compression_model.encode(audio.unsqueeze(0).to(model.device))

                save_path = os.path.join(CACHE_DIR, f"{song_name}_rnd{idx}.pt")
                torch.save({'tokens': tokens.cpu(), 'description': desc}, save_path)
                index_file.append(save_path)

            print(f" -> {song_name} (长音频) 采样了2段: {s1:.1f}s, {s2:.1f}s")

    torch.save(index_file, os.path.join(CACHE_DIR, "index.pt"))
    print(f"完成！共生成 {len(index_file)} 个切片。")


if __name__ == "__main__":
    precompute()