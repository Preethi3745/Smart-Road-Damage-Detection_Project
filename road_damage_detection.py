import cv2
import numpy as np
import argparse
import os


def load_image(image_path):
    """Load an image from disk and stop the program if it's missing."""
    if not os.path.exists(image_path):
        print(f"ERROR: Could not find image '{image_path}'.")
        print("Please place a road image in this folder or use --image path.")
        exit()

    image = cv2.imread(image_path)
    return image


def preprocess_image(image):
    """Convert to grayscale and blur to reduce noise."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray, blurred


def detect_cracks(blurred_image):
    """
    Cracks show up as thin, bright edges on a road.
    We use Canny Edge Detection to highlight them.
    """
    edges = cv2.Canny(blurred_image, 50, 150)

    # Make edges thicker so nearby crack lines connect together
    kernel = np.ones((3, 3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)

    return edges_dilated


def detect_potholes(blurred_image):
    """
    Potholes usually appear as darker, roughly round blobs
    compared to the surrounding road surface.
    We use thresholding to isolate dark regions.
    """
    # Anything darker than this value is considered "possible damage"
    _, thresh = cv2.threshold(blurred_image, 90, 255, cv2.THRESH_BINARY_INV)

    # Clean up small noise specks
    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    return cleaned


def find_damage_contours(mask, min_area=300):
    """Find contours (outlines of blobs) bigger than min_area."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    damage_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    return damage_contours


def draw_results(image, crack_mask, pothole_contours):
    """Draw crack edges in RED and pothole boxes in YELLOW."""
    output = image.copy()

    # Show crack pixels in red
    output[crack_mask > 0] = [0, 0, 255]  # BGR = Red

    # Draw a yellow rectangle + label around each pothole-like blob
    for cnt in pothole_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(output, "Pothole?", (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    return output


def main():
    parser = argparse.ArgumentParser(description="Smart Road Damage Detection (Beginner OpenCV Project)")
    parser.add_argument("--image", type=str, default="sample_road.jpg",
                         help="Path to the road image")
    parser.add_argument("--min_area", type=int, default=300,
                         help="Minimum blob size (pixels) to count as a pothole")
    args = parser.parse_args()

    print("Loading image...")
    image = load_image(args.image)

    print("Preprocessing (grayscale + blur)...")
    gray, blurred = preprocess_image(image)

    print("Detecting cracks (edge detection)...")
    crack_mask = detect_cracks(blurred)

    print("Detecting potholes (dark blob detection)...")
    pothole_mask = detect_potholes(blurred)
    pothole_contours = find_damage_contours(pothole_mask, min_area=args.min_area)

    print(f"Found {len(pothole_contours)} possible pothole region(s).")

    print("Drawing results...")
    result = draw_results(image, crack_mask, pothole_contours)

    # Save all the output images so you can inspect each processing step
    cv2.imwrite("output_1_gray.jpg", gray)
    cv2.imwrite("output_2_crack_edges.jpg", crack_mask)
    cv2.imwrite("output_3_pothole_mask.jpg", pothole_mask)
    cv2.imwrite("output_4_final_result.jpg", result)

    print("\nDone! Check these files in the current folder:")
    print("  output_1_gray.jpg          -> grayscale image")
    print("  output_2_crack_edges.jpg   -> detected crack edges")
    print("  output_3_pothole_mask.jpg  -> dark blob mask (potholes)")
    print("  output_4_final_result.jpg  -> FINAL image with damage marked")


if __name__ == "__main__":
    main()
