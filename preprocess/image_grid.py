import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
import imageio

# image_grid.py
def sample_frames_imageio(video_path: Path, num_frames: int = 9) -> list[Image.Image]:
    """
    用 imageio 读取视频，然后按帧索引采样
    """
    reader = imageio.get_reader(str(video_path), 'ffmpeg')
    total = reader.count_frames()  # 或者 len(reader)（有时不准）
    if total == 0:
        return []
    print(total)

    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = []
    for idx in indices:
        frame = reader.get_data(int(idx))  # RGB ndarray
        frames.append(Image.fromarray(frame))
    reader.close()
    return frames

def make_grid(frames: list[Image.Image], rows: int = 3, cols: int = 3) -> Image.Image:
    """
    将 frames 按 rows x cols 排列成一个大图
    """
    if len(frames) < rows * cols:
        raise ValueError(f"需要至少 {rows*cols} 帧，实际只有 {len(frames)} 帧")

    w, h = frames[0].size
    grid_img = Image.new('RGB', (cols * w, rows * h))
    for i, frame in enumerate(frames[:rows * cols]):
        r, c = divmod(i, cols)
        grid_img.paste(frame, (c * w, r * h))
    return grid_img


def process_videos(input_dir: Path, output_dir: Path):
    """
    对 input_dir 下的每个视频文件采样 9 帧并生成 3x3 拼图, 保存到 output_dir
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for video_file in input_dir.iterdir():
        if video_file.suffix.lower() not in {'.mp4', '.avi', '.mov', '.mkv'}:
            continue

        try:
            frames = sample_frames_imageio(video_file, 9)
            if len(frames) < 9:
                print(f"跳过 {video_file.name}: 帧数不足")
                continue

            grid = make_grid(frames, 3, 3)
            out_path = output_dir / f"{video_file.stem}.jpg"
            grid.save(out_path, quality=95)
            print(f"已保存: {out_path}")
        except Exception as e:
            print(f"处理 {video_file.name} 失败: {e}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="为每个视频生成 3x3 帧拼图")
    parser.add_argument("--input_dir", type=Path, help="视频所在目录")
    parser.add_argument("--output_dir", type=Path, help="拼图保存目录")
    args = parser.parse_args()

    process_videos(args.input_dir, args.output_dir)