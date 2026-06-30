import os
import torch
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

# --- 配置 ---
MODEL_PATH = r"G:\audiocraft-main\models\musicgen-small"
CKPT_PATH = r"G:\audiocraft-main\outputs\my_finetune\best_model.pt"
OUTPUT_DIR = r"G:\audiocraft-main\outputs\my_music_generation"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("正在加载模型，请稍候...")
    model = MusicGen.get_pretrained(MODEL_PATH, device=device)

    print("正在注入微调权重...")
    model.lm.load_state_dict(torch.load(CKPT_PATH, map_location=device))

    
    # 增加 temperature 和 top_k，让生成结果从“复读机”变成“艺术家”
    model.set_generation_params(
        use_sampling=True,  # 开启随机采样，而非确定性输出
        top_k=250,  # 采样空间更广，允许模型选择更多种类的词汇
        temperature=1.3,  # 关键数值：1.0以上增加随机性和创造力，太高会崩，建议1.0-1.3
        duration=20  # 设定单次生成时长
    )

    print("-" * 50)
    print("模型已就绪 (已开启创造力模式)！输入描述词生成音乐 (输入 'exit' 退出)")
    print("-" * 50)

    count = 1
    while True:
        prompt = input("\n请输入音乐描述词: ")
        if prompt.lower() == 'exit':
            break

        print(f"正在生成: '{prompt}' (正在发挥创造力...)")

        # 开始生成
        wav = model.generate([prompt], progress=True)

        # 保存
        file_path = os.path.join(OUTPUT_DIR, f"gen_{count}")
        # 因为你已经装好 ffmpeg，audio_write 可以直接用了
        audio_write(file_path, wav[0].cpu(), model.sample_rate, strategy="loudness")

        print(f"生成成功！文件已保存至: {file_path}.wav")
        count += 1


if __name__ == "__main__":
    main()
