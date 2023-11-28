from PIL import Image

def make_transparent(img):

    # Convert the image to RGBA (if not already in that mode)
    img = img.convert("RGBA")

    # Get the pixel data
    data = img.getdata()

    # Create a new transparent image with the same size
    new_data = []
    for item in data:
        # If the pixel is white, make it transparent
        if item[:3] == (255, 255, 255):
            new_data.append((255, 255, 255, 0))
        else:
            new_data.append(item)

    # Put the new data back into the image
    img.putdata(new_data)

    # Save the result
    img.save('images', "PNG")

    print(f"Image with transparent foreground saved to {output_path}")

