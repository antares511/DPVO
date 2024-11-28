import sys

sys.path.append("/home/kumaraditya_gupta/DROID-SLAM/droid_slam")

import torch
from .config import cfg
from .dpvo import DPVO as BaseDPVO

from depth_video import DepthVideo
from droid_backend import DroidBackend

class IntegratedDPVO(BaseDPVO):
    def __init__(self, args, depth_video):
        cfg.merge_from_file(args.config)
        cfg.BUFFER_SIZE = args.buffer

        super().__init__(cfg, args.net, args.img_ht, args.img_wd, args.viz)
        self.video = depth_video  # Use DepthVideo instance

    def process_frame(self, tstamp, image, intrinsics):
        # Process the frame first using the original DPVO logic
        if self.n + 1 >= self.N:
            raise Exception("The buffer size is too small. Increase the buffer size.")
        
        # Assume image and intrinsics are properly formatted and on CUDA
        self.__call__(tstamp, image, intrinsics)

        for _ in range(12):
            self.update()

        # Now update the DepthVideo with the new computed states
        with self.video.get_lock():
            self.video.append(
                tstamp,
                image,
                self.poses_[self.n],  # Assuming the current pose is updated by DPVO's call
                None,  # If depth is calculated, replace None with actual depth data
                None,  # Similarly, handle sensed disparity if available
                intrinsics / self.RES  # Normalized intrinsics
            )
        
        # Update DPVO's internal counters
        self.n += 1
        self.m += self.M

    def __call__(self, tstamp, image, intrinsics):
        # Overriding the base class processing to handle CUDA and image adjustments
        image_cuda = (image.float() / 255.0).sub_(0.5).mul_(2.0)  # Normalize and scale image as needed for the network
        super().process_frame(tstamp, image_cuda, intrinsics)


class SLAMManager:
    def __init__(self, dpvo_args, droid_args):
        # Initialize DepthVideo
        self.depth_video = DepthVideo(droid_args.image_size, droid_args.buffer, stereo=droid_args.stereo)

        # Initialize DPVO with the modified class
        self.dpvo = IntegratedDPVO(dpvo_args, self.depth_video)

        # Initialize DroidBackend
        self.backend = DroidBackend(self.dpvo.network, self.depth_video, droid_cfg)  # Assuming network configurations are compatible

    def process_frame(self, tstamp, image, intrinsics):
        # Processing a frame through DPVO
        self.dpvo.process_frame(tstamp, image, intrinsics)

    def run_backend_optimization(self, iterations=12):
        # Run backend optimization for a fixed number of iterations
        for _ in range(iterations):
            self.backend.optimize()

    def run_full_process(self, frame_data):
        # Example frame_data should be a list of tuples (tstamp, image, intrinsics)
        for tstamp, image, intrinsics in frame_data:
            self.process_frame(tstamp, image, intrinsics)

        # Once all frames are processed, run backend optimization
        self.run_backend_optimization()
