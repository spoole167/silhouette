from PIL import Image
import cv2
import numpy as np
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process an image to generate an SVG file.")

    # Add arguments
    parser.add_argument("input_image", type=str, help="Path to the input PNG image.")
    parser.add_argument("output_svg", type=str, help="Path to the output SVG file.")
    parser.add_argument(
        "--invert",
        action="store_true",
        help="If set, inverts the image before processing"
    )

    return parser.parse_args()


# Function to convert contours to SVG path data
def contours_to_svg_path(contours, epsilon_factor=0.001):
    svg_path = ""
    for contour in contours:
        if len(contour) < 2:  # Skip small artifacts
            continue
        # Simplify contour using approxPolyDP
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        simplified_contour = cv2.approxPolyDP(contour, epsilon, True)

        path_data = "M "
        for point in simplified_contour:
            x, y = point[0]
            path_data += f"{x} {y} "
        path_data += "Z "  # Close the path
        svg_path += path_data + "\n"
    return svg_path


# Function to generate an SVG file
def generate_svg(svg_path_data, width, height, output_file):
    svg_content = (
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' "
        f"viewBox='0 0 {width} {height}'>\n"
    )
    svg_content += f"<path d='{svg_path_data}' fill='none' stroke='black' stroke-width='1'/>\n"
    svg_content += "</svg>"

    with open(output_file, "w") as f:
        f.write(svg_content)


# Main function to process the image
def png_to_svg(image_path, output_svg, invert):
    # Load the image
    img = Image.open(image_path).convert("L")
    width, height = img.size

    # Convert image to binary (black and white)
    img_array = np.array(img)
    if invert:
        img_array = cv2.bitwise_not(img_array)

    # Apply Gaussian blur to smooth edges
    blurred_img = cv2.GaussianBlur(img_array, (15, 15), 0)

    # Threshold the image
    _, binary_img = cv2.threshold(blurred_img, 127, 255, cv2.THRESH_BINARY)

    # Find contours for white objects (255)
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convert contours to SVG path data with simplification
    svg_path_data = contours_to_svg_path(contours)

    # Generate the SVG file
    generate_svg(svg_path_data, width, height, output_svg)


# Example usage
if __name__ == "__main__":
    args = parse_arguments()
    input_image_path = args.input_image
    output_svg_path = args.output_svg
    invert = args.invert

    png_to_svg(input_image_path,output_svg_path,invert)
