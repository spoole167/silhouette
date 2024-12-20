import pygame
import numpy as np
import sys
import argparse
import svgwrite
from PIL import Image, ImageOps
from scipy.ndimage import binary_dilation, binary_erosion, binary_fill_holes
import cairosvg
import cv2
import os
import io

def BandW(image):
    # Convert image to a NumPy array
    image_array = np.array(image)

    # Create a mask for non-black pixels
    non_black_mask = (image_array[:, :, 0] != 0) | (image_array[:, :, 1] != 0) | (image_array[:, :, 2] != 0)

    # Set non-black pixels to white
    image_array[non_black_mask] = [255, 255, 255]

    # Convert back to a PIL image
    return Image.fromarray(image_array)

def g2Pillow(surface):
    # Ensure the surface is in RGB format
    surface_rgb = pygame.Surface.convert(surface)
    width, height = surface_rgb.get_size()

    # Convert pygame.Surface to raw string data
    raw_str = pygame.image.tostring(surface_rgb, 'RGB')

    # Create a Pillow image from raw string data
    pil_image = Image.frombytes('RGB', (width, height), raw_str)
    return pil_image

def svg2image(dwg):
    svg_content = dwg.tostring()
    image_bytes = io.BytesIO()
    cairosvg.svg2png(bytestring=svg_content.encode('utf-8'), write_to=image_bytes)
    image_bytes.seek(0)
    return Image.open(image_bytes)

def add_dimensions_to_svg(svg_content, width="500px", height="500px"):
    if 'width' not in svg_content or 'height' not in svg_content:
        svg_content = svg_content.replace(
            '<svg', f'<svg width="{width}" height="{height}"', 1
        )
    return svg_content

def image2svg(pil_image,output_svg_path):


    # Load the image
    # pil_image = pil_image.convert('RGB')

    # Convert to NumPy array
    numpy_image = np.array(pil_image)

    # Convert from RGB to BGR for OpenCV
    numpy_image_bgr = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

    # Convert to grayscale
    gray = cv2.cvtColor(numpy_image_bgr, cv2.COLOR_BGR2GRAY)

    # Threshold the image
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Create an SVG drawing
    dwg = svgwrite.Drawing(output_svg_path, profile='tiny',size=('500px', '500px'))


    # Add contours to SVG
    for contour in contours:
        # Simplify contour for smoothness
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Convert contour points to SVG path format
        path_data = "M " + " ".join([f"{point[0][0]},{point[0][1]}" for point in approx]) + " Z"

        # Add the path to the SVG
        dwg.add(dwg.path(d=path_data, fill='none', stroke='red', stroke_width=2))

    # Save the SVG file
    dwg.save()
    print("svg saved to "+output_svg_path)
    return dwg

def generate_unique_filename(directory, filename):
    """
    Generates a unique filename in the given directory. If the file exists,
    it appends a numeric suffix to create a unique name.

    Args:
        directory (str): Path to the directory to check.
        filename (str): Desired filename.

    Returns:
        str: Unique filename with path.
    """
    base, extension = os.path.splitext(filename)
    counter = 1
    unique_filename = filename

    while os.path.exists(os.path.join(directory, unique_filename)):
        unique_filename = f"{base}_{counter}{extension}"
        counter += 1

    return os.path.join(directory, unique_filename)

# Example usage:
directory = "./"  # Replace with your directory path
filename = "example.txt"

unique_file = generate_unique_filename(directory, filename)
print(f"Unique filename: {unique_file}")


def remove(image, mask, color=(0, 255, 0)):


    image_array = pygame.surfarray.array3d(image).copy()
    image_array[mask] = color
    return pygame.surfarray.make_surface(image_array)


def save_mask_as_image(mask, save_path):
    """
    Save a binary mask as a grayscale image.
    :param mask: 2D binary NumPy array representing the mask.
    :param save_path: File path to save the mask image.
    """
    # Convert the mask to uint8 format (0 and 255)
    mask_image = (mask * 255).astype(np.uint8)

    # Create a Pillow image from the array
    image = Image.fromarray(mask_image)

    # Save the image
    image.save(save_path)
    print(f"Mask saved as image to {save_path}")

def extract_boundaries_as_lines(mask):
    """Extract boundaries of the binary mask as a continuous line."""
    dilated_mask = binary_dilation(mask)
    boundary_mask = dilated_mask ^ mask  # XOR to isolate the boundary
    boundary_coords = np.argwhere(boundary_mask)

    # Convert to (x, y) format for compatibility with drawing
    boundary_coords = [(x, y) for y, x in boundary_coords]

    if not boundary_coords:
        print("No boundary coordinates found.")

    # Trace the boundary
    line = []
    visited = set()
    current = boundary_coords[0] if boundary_coords else None

    while current:
        line.append(current)
        visited.add(current)

        # Find neighbors
        neighbors = [
            (current[0] + dx, current[1] + dy)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
        ]

        # Select the next pixel
        current = next(
            (neighbor for neighbor in neighbors if neighbor in boundary_coords and neighbor not in visited),
            None,
        )

    print(f"Extracted boundary line: {line}")
    return line




def scale_surface(surface, new_width, new_height):
    """Scale a pygame surface to a new size."""
    return pygame.transform.smoothscale(surface, (new_width, new_height))

def highlight_color(image, base_color, tolerance):
    """Highlight pixels within the specified tolerance of the base color."""
    image_array = pygame.surfarray.array3d(image)
    mask = np.all(np.abs(image_array - base_color) <= tolerance, axis=2)
    highlighted = np.zeros_like(image_array)
    highlighted[mask] = [255, 0, 0]  # Highlight in red
    highlighted_surface = pygame.surfarray.make_surface(highlighted)
    return highlighted_surface, mask

def process_mask(mask, remove_noise=True, fill_gaps=True):
    """Process the mask to remove noise and fill gaps."""
    processed_mask = mask.copy()
    if remove_noise:
        processed_mask = binary_erosion(processed_mask, iterations=2)
        processed_mask = binary_dilation(processed_mask, iterations=2)
    if fill_gaps:
        processed_mask = binary_fill_holes(processed_mask)
    return processed_mask


def save_image(image, mask, save_path):
    """Save the highlighted image to disk."""
    image_array = pygame.surfarray.array3d(image)
    processed_image = np.zeros_like(image_array)

    if np.any(mask):  # Check if there are any True values in the mask
        avg_color = image_array[mask].mean(axis=0).astype(np.uint8)
    else:
        avg_color = [0, 0, 0]  # Default to black if no pixels are masked

    # Create a new mask array initialized with zeros
    height, width, _ = image_array.shape
    new_mask = np.zeros((height, width, 3), dtype=np.uint8)

    # Set all masked pixels to the average color
    new_mask[mask] = avg_color

    processed_image[mask] = [255,255,255] #image_array[mask]  # Retain only highlighted areas

    image2 = Image.fromarray(processed_image)
    # Save the image to a file
    image2.save(save_path)
    #pygame.image.save(processed_image, save_path)
    print(f"Saved highlighted image to {save_path}")
    return processed_image

def average_mask(image_array, mask):
    """
    Update the mask by setting all its pixels to the average color of the masked area.

    Args:
        image_array (numpy.ndarray): 3D array (H, W, 3) representing the image.
        mask (numpy.ndarray): 2D boolean array (H, W) where True indicates masked pixels.

    Returns:
        numpy.ndarray: New 3D mask array with masked pixels set to the average color.
    """
    # Ensure the input is a NumPy array
    image_array = np.asarray(image_array)

    # Calculate the average color of the masked pixels
    if np.any(mask):  # Check if there are any True values in the mask
        avg_color = image_array[mask].mean(axis=0).astype(np.uint8)
    else:
        avg_color = [0, 0, 0]  # Default to black if no pixels are masked

    # Create a new mask array initialized with zeros
    height, width, _ = image_array.shape
    new_mask = np.zeros((height, width, 3), dtype=np.uint8)

    # Set all masked pixels to the average color
    new_mask[mask] = avg_color

    return new_mask

def main(image_path):

    pygame.init()
    screen_info = pygame.display.Info()
    screen_width = screen_info.current_w
    screen_height = screen_info.current_h


    # Load the image
    image = pygame.image.load(image_path)
    result_image=image
    original_width, original_height = image.get_size()

    # Determine the screen size

    # Scale to fit within the screen
    scale_factor = min(screen_width / (4 * original_width), screen_height / original_height)
    scaled_width = int(original_width * scale_factor)
    scaled_height = int(original_height * scale_factor)

    # Set up the display for three panels
    display_width = scaled_width * 4
    display_height = scaled_height
    screen = pygame.display.set_mode((display_width, display_height))

    pygame.display.set_caption("Interactive Color Highlighter")

    # Scale the original image
    scaled_image = scale_surface(image, scaled_width, scaled_height)

    # Variables
    tolerance = 10
    selected_color = None
    highlighted_surface = None
    svg_surface = None
    modified_surface =None
    mask = None

    # Main loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                if x < scaled_width:  # Only select pixels in the left panel
                    orig_x = int(x / scale_factor)
                    orig_y = int(y / scale_factor)
                    selected_color = image.get_at((orig_x, orig_y))[:3]
                    print(f"Selected color: {selected_color}")
                    highlighted_surface, mask = highlight_color(image, selected_color, tolerance)
                    highlighted_surface = scale_surface(highlighted_surface, scaled_width, scaled_height)
                    boundary_surface = None  # Reset boundary surface
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:  # Increase tolerance
                    tolerance = min(tolerance + 1, 255)
                    print(f"Tolerance increased to {tolerance}")
                    if selected_color:
                        highlighted_surface, mask = highlight_color(image, selected_color, tolerance)
                        highlighted_surface = scale_surface(highlighted_surface, scaled_width, scaled_height)
                        svg_surface = None  # Reset boundary surface
                elif event.key == pygame.K_DOWN:  # Decrease tolerance
                    tolerance = max(tolerance - 1, 0)
                    print(f"Tolerance decreased to {tolerance}")
                    if selected_color:
                        highlighted_surface, mask = highlight_color(image, selected_color, tolerance)
                        highlighted_surface = scale_surface(highlighted_surface, scaled_width, scaled_height)
                        svg_surface = None  # Reset boundary surface
                elif event.key == pygame.K_p:  # Process mask
                    if mask is not None:
                        print("Processing mask to remove noise and fill gaps...")
                        mask = process_mask(mask)

                        highlighted_color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
                        highlighted_color[mask] = [255, 0, 0]
                        highlighted_surface = pygame.surfarray.make_surface(highlighted_color)
                        highlighted_surface = scale_surface(highlighted_surface, scaled_width, scaled_height)
                        svg_surface = None  # Reset boundary surface
                elif event.key == pygame.K_s:
                    if mask is not None:
                        f=generate_unique_filename("output","layer.png")

                        pi=save_image(image, mask, f)
                        f=generate_unique_filename("output","mask.svg")
                        dwg=image2svg(pi,f)
                        #pi=svg2image(dwg)
                        #pi=pygame.surfarray.array3d(pi)
                        #svg_surface=pi#pygame.surfarray.make_surface(pi)
                        #svg_surface = scale_surface(svg_surface, scaled_width, scaled_height)

                elif event.key == pygame.K_a:  # Apply mask and display the modified image in the fourth panel
                    if mask is not None:
                        print("Removing mask")
                        result_image = remove(result_image, mask,selected_color)
                        modified_surface = scale_surface(result_image, scaled_width, scaled_height)


                elif event.key == pygame.K_b:  # Add boundary lines
                    if mask is not None:
                        print("Adding boundary lines...")



        # Draw the original image on the left panel
        screen.blit(scaled_image, (0, 0))

        # Draw the highlighted image on the middle panel
        if highlighted_surface:
            screen.blit(highlighted_surface, (scaled_width, 0))

        # Draw the image with boundary lines on the right panel
        if svg_surface:
            screen.blit(svg_surface, (2 * scaled_width, 0))
        if modified_surface:
            screen.blit(modified_surface, (3 * scaled_width, 0))

        # Update the display
        pygame.display.flip()

    # Quit pygame
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive Color Highlighter")
    parser.add_argument("image_path", help="Path to the image file")
    args = parser.parse_args()

    try:
        main(args.image_path)
    except FileNotFoundError:
        print(f"Error: File '{args.image_path}' not found.")
        sys.exit(1)
