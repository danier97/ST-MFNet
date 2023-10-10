# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import math
import time
from types import SimpleNamespace
from typing import Iterator
import os
import subprocess

import cv2
import torch
import torchvision.transforms.functional as TF
from tqdm import tqdm
from cog import BasePredictor, Input, Path

from utility import tensor2rgb
from models.stmfnet import STMFNet

STMFNET_WEIGHTS_URL = "https://weights.replicate.delivery/default/stmfnet/stmfnet.pth"
STMFNET_WEIGHTS_PATH = "stmfnet.pth"


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    try:
        subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    except subprocess.CalledProcessError:
        print("Extraction with -x failed. Trying download without extraction...")
        subprocess.check_call(["pget", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def setup(self):
        """
        Set up the prediction environment.

        This method initializes the model, its parameters, and the GPU device for computation.
        It also loads the STMFNet model using the specified checkpoint.
        Lastly, it ensures there is an output directory for storing the enhanced videos.
        """
        if not os.path.exists(STMFNET_WEIGHTS_PATH):
            download_weights(STMFNET_WEIGHTS_URL, STMFNET_WEIGHTS_PATH)

        args = SimpleNamespace(
            **{
                "gpu_id": (gpu_id := 0),
                "net": (net := "STMFNet"),
                "checkpoint": (checkpoint := STMFNET_WEIGHTS_PATH),
                "size": (size := "1920x1080"),
                "patch_size": (patch_size := None),
                "overlap": (overlap := None),
                "batch_size": (batch_size := None),
                "out_fps": (out_fps := 144),
                "out_dir": (out_dir := "."),
                "featc": (featc := [64, 128, 256, 512]),
                "featnet": (featnet := "UMultiScaleResNext"),
                "featnorm": (featnorm := "batch"),
                "kernel_size": (kernel_size := 5),
                "dilation": (dilation := 1),
                "finetune_pwc": (finetune_pwc := False),
            }
        )
        torch.cuda.set_device(gpu_id)

        self.net = net
        self.size = size
        self.model = STMFNet(args).cuda()
        print("Loading the model...")
        checkpoint = torch.load(checkpoint)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

    def predict(
        self,
        mp4: Path = Input(description="Upload an mp4 video file."),
        framerate_multiplier: int = Input(
            description="Determines how many intermediate frames to generate between original frames. E.g., a value of 2 will double the frame rate, and 4 will quadruple it, etc.",
            default=2,
            choices=[2, 4, 8, 16, 32],
        ),
        keep_original_duration: bool = Input(
            description="Should the enhanced video retain the original duration? If set to `True`, the model will adjust the frame rate to maintain the video's original duration after adding interpolated frames. If set to `False`, the frame rate will be set based on `custom_fps`.",
            default=True,
        ),
        custom_fps: float = Input(
            description="Set `keep_original_duration` to `False` to use this! Desired frame rate (fps) for the enhanced video. This will only be considered if `keep_original_duration` is set to `False`.",
            default=None,
            ge=1,
            le=240,
        ),
    ) -> Iterator[Path]:
        """
        Enhance a video by increasing its frame rate using frame interpolation.

        Parameters:
        - mp4 (Path): Path to the video file.
        - keep_original_duration (bool): Indicator to maintain the original video duration after frame interpolation.
        - custom_fps (float): Target frame rate for the enhanced video when not maintaining the original duration.
        - framerate_multiplier (int): Multiplier for the number of frames.

        Returns:
        Iterator[Path]: Paths to the generated enhanced video files.
        """

        num_iterations = int(math.log2(framerate_multiplier))
        original_seq_name = os.path.basename(mp4).split(".")[0]

        for enhancing_iteration in tqdm(range(num_iterations), desc="Enhancing iterations"):
            # Opening the video and extracting essential properties
            video = cv2.VideoCapture(str(mp4))
            original_video_fps = video.get(cv2.CAP_PROP_FPS)
            width, height = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            original_num_frames = sum(video.read()[0] for _ in range(int(video.get(cv2.CAP_PROP_FRAME_COUNT))))

            # Informing the user of video details before processing
            print(f"Video Name: {original_seq_name}")
            print(f"Original Frame Rate (FPS): {original_video_fps}")
            print(f"Original Total Number of Frames: {original_num_frames}")

            img_array = []
            # Processing each set of 4 frames for frame rate enhancement
            for t in tqdm(range(0, original_num_frames - 3), desc="Processing frames"):
                video.set(cv2.CAP_PROP_POS_FRAMES, t)
                _, rawFrame0 = video.read()
                _, rawFrame1 = video.read()
                _, rawFrame2 = video.read()
                _, rawFrame3 = video.read()

                # If any frame in the set of 4 is missing, stop processing
                if any(frame is None for frame in [rawFrame0, rawFrame1, rawFrame2, rawFrame3]):
                    break

                # Convert frames to tensors and move them to GPU
                frame0 = TF.to_tensor(rawFrame0)[None, ...].cuda()
                frame1 = TF.to_tensor(rawFrame1)[None, ...].cuda()
                frame2 = TF.to_tensor(rawFrame2)[None, ...].cuda()
                frame3 = TF.to_tensor(rawFrame3)[None, ...].cuda()

                # Use the trained model to predict enhanced frames
                with torch.no_grad():
                    out = self.model(frame0, frame1, frame2, frame3)

                # Special handling for the very first
                if t == 0:
                    img_array += [tensor2rgb(frame0)[0]] * 2 + [tensor2rgb(frame1)[0]]

                img_array += [tensor2rgb(out)[0], tensor2rgb(frame2)[0]]

                # Special handling for the last sets of frames
                if t == original_num_frames - 4:
                    img_array += [tensor2rgb(frame3)[0]] * 2
            video.release()

            # Decide the output video's fps
            new_num_frames = len(img_array)
            output_fps = (
                new_num_frames * original_video_fps
            ) / original_num_frames  # Compute the fps that keeps video playback constant (duration of video)
            if (not keep_original_duration) and (custom_fps is not None) and (custom_fps >= 1):
                output_fps = custom_fps

            # Create and write frames to the output video
            avi_outname = f"{original_seq_name}_{enhancing_iteration}.avi"
            new_num_frames = len(img_array)

            print(f"Output filename: {avi_outname}")
            print(f"New Total Number of Frames: {new_num_frames}")

            cv2writer = cv2.VideoWriter(
                avi_outname,
                cv2.VideoWriter_fourcc(*"DIVX"),  # NOTE: codec issues mean we have to export as avi using DIVX
                output_fps,
                (width, height),
            )

            for frame in img_array:
                cv2writer.write(frame)
            cv2writer.release()

            # Convert the AVI video to MP4 format using ffmpeg (NOTE: We use ffmpeg because we have codec issues with cv2 and mp4)
            mp4_outname = avi_outname.replace(".avi", ".mp4")
            cmd = ["ffmpeg", "-i", avi_outname, mp4_outname, "-y"]
            subprocess.run(cmd)

            # Append the output path and prepare for the next iteration if needed
            mp4 = mp4_outname
            yield Path(mp4_outname)
