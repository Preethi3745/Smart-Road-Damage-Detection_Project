# Smart Road Damage Detection (Beginner OpenCV Project)

A simple Python project that detects **cracks** and **potholes** in road
images using only classic OpenCV image-processing — no machine learning,
no complicated code.

## How it works

| Step | What happens |
|------|--------------|
| 1. Load image | Read the road photo with OpenCV |
| 2. Grayscale + blur | Simplify the image and remove noise |
| 3. Crack detection | Canny edge detection finds thin crack lines |
| 4. Pothole detection | Thresholding finds dark, blob-shaped regions |
| 5. Draw results | Cracks shown in red, potholes boxed in yellow |

## Setup

```bash
pip install -r requirements.txt
```

## Run it

```bash
python road_damage_detection.py --image your_road_photo.jpg
```

If you don't have a photo handy, the script looks for `sample_road.jpg`
by default — just drop any road image with that name in the folder.

### Optional setting
- `--min_area` — minimum blob size (in pixels) to count as a pothole
  (default `300`). Raise this number if it's detecting too many small
  spots; lower it to catch smaller potholes.

## Output files

Running the script creates four images so you can see every step:

- `output_1_gray.jpg` — grayscale version
- `output_2_crack_edges.jpg` — detected crack edges
- `output_3_pothole_mask.jpg` — dark-blob mask used for potholes
- `output_4_final_result.jpg` — **final image** with damage marked

## Notes for learning

This uses "classical" computer vision (edges + thresholds + contours),
which is perfect for understanding the basics before moving on to
deep-learning-based detectors (like YOLO) for a more advanced version.
Tips to improve accuracy:
- Adjust the `50, 150` values in `cv2.Canny()` for more/less sensitive
  crack detection.
- Adjust the `90` value in `cv2.threshold()` depending on how dark your
  road photos are.
