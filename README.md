# TouhouVideoFeatureExtractor
High performance video feature extractor using CLIP \
Video is first segmented into shots using TransNetV2, for each shot a signle frames in the middle of shot is used for CLIP feature extraction if duration of the shot is less than 2 seconds, other wise 3 frames from the beginning, middle and end of the shot is used.
