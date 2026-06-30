import os
import json
import torchaudio

DATA_DIR = "G:/music_train_data"
OUTPUT_DIR = "G:/music_train_data/data"

DESCRIPTIONS = {
    "All time low.wav": "另类流行，独特男声，口哨旋律，热带浩室节拍，丰富合成器音色，忧郁但律动感强，现代感。",
    "always online.wav": "早期温暖电子R&B流行，明亮合成器拨弦，丝滑少年男声，温柔转音，异地恋，温暖守护感，安心氛围。",
    "baby.wav": "2010年代青少年流行，轻快R&B，合成器流行，洗脑副歌，说唱桥段，青春甜美，中速节奏。",
    "call me maybe.wav": "泡泡糖流行，甜美女声，抓耳吉他riff，合成器流行，魔性洗脑旋律，欢快活泼，夏日氛围，快节奏。",
    "exile.wav": "欧美独立民谣另类流行，清冷钢琴搭配空旷氛围弦乐，男女声交替和声拉扯，分手多年重逢，隔阂无法和解，破碎疏离情感。",
    "hero.wav": "电子舞曲抒情，力量感女声，情绪饱满，钢琴前奏，重拍drop，史诗感，颂歌式编曲，充满希望。",
    "letting go.wav": "慵懒都市爵士抒情，极简钢琴律动，松弛烟嗓女声，释怀放手，无奈接纳，轻盈节奏。",
    "stay.wav": "现代流行朋克，快节奏，强劲鼓点，青春叛逆又恳切的嗓音，记忆点十足的副歌。",
    "stronger.wav": "励志流行摇滚，驱动感节奏，鼓组与电吉他，充满力量与重生感的女声。",
    "try.wav": "原声流行民谣，清澈女声，简单吉他扫弦，温暖治愈，自我接纳，轻盈节奏，纯粹干净。",
    "whataya want from me.wav": "强力流行摇滚，爆发力男声，宽广音域，戏剧化编曲，电吉他solo，张力十足的情歌。",
    "学不会.wav": "欧陆风情华语抒情大作，清冷钢琴与宏大弦乐交织，林俊杰宣泄式撕裂高音，情感层层递进爆发，爱情里永远学不会放手的极致痛楚与挣扎。",
    "雨过后的风景.wav": "怀旧治愈系华语流行抒情，温柔细腻女声，舒缓钢琴配器搭配轻盈节奏，雨后清新氛围感，挫折后自我修复、与遗憾共存的释然与温暖。",
    "病变.wav": "暗黑电子流行，微哑戏剧化女声，强烈节奏，突出合成器音色，病态美感，情绪爆发。",
    "曹操.wav": "中式史诗摇滚流行，失真电吉他+强劲鼓点+传统打击乐，清亮金属感高音男声，英雄乱世史诗感。",
    "孤独患者.wav": "华语流行抒情，情绪化男声，钢琴与弦乐，忧郁深沉，慢速节奏。",
    "光年之外.wav": "史诗太空抒情流行，钢琴+宏大弦乐+轻电子，铁肺力量感女声，星际绝望与不屈浪漫。",
    "绝对占有 相对自由.wav": "暗黑迷幻独立民谣，低频贝斯+氛围电子，中性慵懒缠绵女声，偏执占有欲爱情。",
    "恋人.wav": "极简复古叙事抒情，钢琴+轻柔吉他，沙哑低沉男声，日常碎片回忆，未化解的温柔遗憾。",
    "模特.wav": "现代布鲁斯流行，慵懒男声，清音电吉他，极简编曲，复古氛围，律动贝斯线。",
    "起风了.wav": "日系治愈华语民谣，柔和木吉他+风铃合成器，空灵细腻嗓音，青春追梦，怀旧释然。",
    "晴天.wav": "校园复古华语抒情流行，清新木吉他前奏配温柔弦乐，青涩怀旧少年男声，青春校园无疾而终的初恋遗憾与夏日回忆。",
    "淘汰.wav": "粤语经典苦情抒情，简约钢琴与弦乐，温暖醇厚男中音，三视角冲突，隐忍遗憾，克制痛感。",
    "唯一.wav": "宏大流行抒情，邓紫棋标志性高音与情绪爆发，钢琴与弦乐渐强铺垫。",
    "我的歌声里.wav": "清新治愈流行民谣，温柔纯净女声，简单旋律，极强感染力，温暖抚慰人心。",
    "喜欢你.wav": "现代改编粤语甜系流行，柔和复古爵士编曲，弱化力量感甜美女声，懵懂心动，简单心跳感。",
    "小半.wav": "温柔民谣流行，陈粒标志性气声唱法，流畅旋律，简约但情绪层次丰富的编曲。",
    "虚拟.wav": "独立民谣风格，空灵慵懒略带迷幻女声，吉他琶音，都市孤独感与诗意叙事。",
    "演员.wav": "都市情歌经典，薛之谦标志性“苦情歌”风格，抓耳旋律，钢琴与弦乐主导。",
    "一天一天.wav": "经典韩式R&B抒情，深情厚重男声，慢速节奏，旋律悠扬且紧绷，怀念与无力感。",
    "遗失的心跳.wav": "都市电子舞曲抒情，动感合成器鼓组+悲伤旋律，酷飒姐姐中音，分手后空虚与心碎。",
    "意外.wav": "英伦摇滚苦情华语流行，冰冷钢琴前奏，副歌弦乐与鼓点爆发，沙哑哽咽男声，危险致命爱情遗憾。",
    "阴天.wav": "复古都市爵士抒情，慵懒爵士钢琴，低沉磁性烟嗓女声，雨天独处，激情褪去，迷茫与清醒交织。",
}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "data.jsonl")
    records = []

    for filename in sorted(os.listdir(DATA_DIR)):
        if not filename.endswith('.wav'):
            continue
        fpath = os.path.join(DATA_DIR, filename).replace("\\", "/")
        desc = DESCRIPTIONS.get(filename)
        if desc is None:
            continue
        try:
            info = torchaudio.info(fpath)
            total_duration = info.num_frames / info.sample_rate

            skip_seconds = 3.0
            actual_start = min(skip_seconds, total_duration)
            remaining_duration = max(0.0, total_duration - actual_start)

            if remaining_duration <= 0:
                continue

            records.append({
                "path": fpath,
                "duration": round(min(remaining_duration, 30.0), 2),
                "start_time": round(actual_start, 2),
                "sample_rate": info.sample_rate,
                "description": desc
            })
        except Exception:
            continue

    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()