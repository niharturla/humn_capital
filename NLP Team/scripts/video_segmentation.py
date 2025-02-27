# Video Segmentation
# Author: Kevin Huang

from moviepy.editor import VideoFileClip

def video_segmentation(video_path, start_time, end_time, output_path):
    video = VideoFileClip(video_path)
    video = video.subclip(start_time, end_time)
    video.write_videofile(output_path)

# time from 41:20 to 45:57
video_segmentation("Mock 1-on-1s - Recording.mp4", 41*60+20, 45*60+57, "segmented_video1.mp4")