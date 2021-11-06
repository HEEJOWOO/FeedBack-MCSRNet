
import argparse
import os
import cv2
import numpy as np
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from utils import *
from espcn_pytorch import ESPCN
from models import MCSRNet

parser = argparse.ArgumentParser(description="make the noise image ")
parser.add_argument("--file", type=str, required=True,
                    help="Test low resolution video name.")
parser.add_argument("--scale-factor", type=int, required=False, choices=[2, 3, 4, 8],
                    help="Super resolution upscale factor. (default:4)")
parser.add_argument("--view", action="store_true",
                    help="real time to show.")
parser.add_argument("--cuda", action="store_true",
                    help="Enables cuda")
parser.add_argument("--test_noiseL", type=float, default=25, help='noise level used on test set')

args = parser.parse_args()
print(args)


# img preprocessing operation
pil2tensor = transforms.ToTensor()
tensor2pil = transforms.ToPILImage()

# Open video file
video_name = args.file
print(f"Reading `{os.path.basename(video_name)}`...")
video_capture = cv2.VideoCapture(video_name)
# Prepare to write the processed image into the video.
fps = video_capture.get(cv2.CAP_PROP_FPS)
total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
# Set video size
size = (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# Video write loader.
video_writer = cv2.VideoWriter(f"noise_{os.path.basename(video_name)}",cv2.VideoWriter_fourcc(*"MPEG"), fps, size)
# read frame
success, raw_frame = video_capture.read()
progress_bar = tqdm(range(total_frames), desc="[processing video and saving/view result videos]")
for index in progress_bar:
    if success:
        img = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(np.array(img).astype(np.float32).transpose([2, 0, 1]), 0) / 255.0
        ISource = torch.from_numpy(img)
        noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=args.test_noiseL / 255.)
        INoisy = (ISource + noise).squeeze(0)
        noise_frame = cv2.cvtColor(np.asarray(denormalize(INoisy).permute(1, 2, 0).byte().cpu().numpy()), cv2.COLOR_RGB2BGR)
        # save sr video
        video_writer.write(noise_frame)
        if args.view:
            # display video
            cv2.imshow("LR video to Noise LR video ", final_image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        # next frame
        success, raw_frame = video_capture.read()
