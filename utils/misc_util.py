import os
import torch
import random
import numpy as np
# import logging
from loguru import logger


def fix_seeds(random_seed=42, use_deterministic_algorithms=False):
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
        
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False    
    
    if use_deterministic_algorithms:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.use_deterministic_algorithms(use_deterministic_algorithms)

def get_class(kls):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m

def setup_logging(log_file='my_log.log'):
    logger.add(log_file, format="{time:YYYY-MM-DD HH:mm:ss} {file}:{line} {level} {message}", level="DEBUG")

# def build_logger(mode, out_dir=None):
#     logger = logging.getLogger(__name__)
#     logger.setLevel(logging.INFO)
#     formatter = logging.Formatter(
#         fmt='%(asctime)s %(filename)s:%(lineno)d %(message)s',
#         datefmt='%Y-%m-%d %H:%M:%S'
#     )

#     stream_handler = logging.StreamHandler()
#     stream_handler.setLevel(logging.INFO)
#     stream_handler.setFormatter(formatter)
#     logger.addHandler(stream_handler)

#     if out_dir is not None:
#         file_handler = logging.FileHandler(os.path.join(out_dir, f'{mode}_log.log'))
#         file_handler.setLevel(logging.INFO)
#         file_handler.setFormatter(formatter)
#         logger.addHandler(file_handler)
    
#     return logger


def is_video_file(filepath):
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.mpeg', '.mpg', '.m4v', '.3gp')
    return filepath.lower().endswith(video_extensions)

def save_frames_from_video(video_path, output_dir, step=10):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = os.path.join(output_dir, f"frame_{frame_count:05d}.jpg")
        if frame_count % step == 0:
            cv2.imwrite(frame_filename, frame)
        frame_count += 1
    cap.release()
    print(f"Total frames saved: {frame_count}")

def save_video(image_path):
    images = [img for img in os.listdir(image_path) if img.endswith(".png") or img.endswith(".jpg")]
    # for image in images:
    #     img_name = image.split('.')[0]
    #     img_name_id = img_name[6:].zfill(5)
    #     os.rename(os.path.join(image_path, image), os.path.join(image_path, f'frame_{img_name_id}.jpg'))
    images = sorted(images)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use 'XVID', 'MJPG', etc. depending on your needs
    output_video = os.path.join(image_path, '../video.mp4',)
    fps = 30
    frame = cv2.imread(os.path.join(image_path, images[0]))
    height, width = frame.shape[:2]
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    for image in tqdm(images, total=len(images)):
        frame = cv2.imread(os.path.join(image_path, image))
        video_writer.write(frame)
    video_writer.release()
    print(f"Video saved as {output_video}")