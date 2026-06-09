#!/usr/bin/env python3
"""
PhishGuard Icon Generator
=========================
Generates placeholder PNG icons for the Chrome extension.
Run this script to create the icons directory with proper PNG files.
"""

import os
import base64

# Minimal 1x1 transparent PNG (base64)
TRANSPARENT_PNG_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="


def create_simple_icon(size: int, output_path: str):
    """
    Create a simple colored icon using PIL if available.

    Falls back to transparent PNG if PIL is not installed.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont

        # Create image with gradient background
        img = Image.new('RGBA', (size, size), (26, 118, 210, 255))  # Blue
        draw = ImageDraw.Draw(img)

        # Draw a simple shield shape
        center = size // 2
        margin = size // 6

        # Shield outline
        points = [
            (center, margin),  # Top
            (size - margin, margin + size // 6),  # Top right
            (size - margin, size - margin - size // 4),  # Bottom right
            (center, size - margin),  # Bottom
            (margin, size - margin - size // 4),  # Bottom left
            (margin, margin + size // 6),  # Top left
        ]

        # Draw filled shield
        draw.polygon(points, fill=(255, 255, 255, 255))

        # Draw "P" letter
        letter_size = size // 2
        letter_margin = center - letter_size // 2
        draw.ellipse([
            letter_margin + letter_size // 4,
            letter_margin,
            letter_margin + letter_size - letter_size // 4,
            letter_margin + letter_size
        ], fill=(26, 118, 210, 255))

        # Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img.save(output_path, 'PNG')
        print(f"Created icon: {output_path}")

    except ImportError:
        print("PIL not available. Creating transparent placeholder icons.")
        create_transparent_icon(size, output_path)


def create_transparent_icon(size: int, output_path: str):
    """Create a transparent PNG as placeholder."""
    import zlib
    import struct

    def create_png(width, height, color=(100, 100, 100, 200)):
        def png_chunk(chunk_type, data):
            chunk_data = chunk_type + data
            return struct.pack('>I', len(data)) + chunk_data + struct.pack('>I', zlib.crc32(chunk_data) & 0xffffffff)

        # PNG signature
        signature = b'\x89PNG\r\n\x1a\n'

        # IHDR chunk
        ihdr_data = struct.pack('>IIBBBBB', width, height, 8, 6, 0, 0, 0)
        ihdr = png_chunk(b'IHDR', ihdr_data)

        # IDAT chunk (image data)
        raw_data = b''
        for y in range(height):
            raw_data += b'\x00'  # Filter byte
            for x in range(width):
                # Simple circle
                cx, cy = width // 2, height // 2
                r = min(width, height) // 3
                dx, dy = x - cx, y - cy
                if dx * dx + dy * dy < r * r:
                    raw_data += bytes([26, 118, 210, 255])  # Blue
                else:
                    raw_data += bytes([0, 0, 0, 0])  # Transparent

        compressed = zlib.compress(raw_data, 9)
        idat = png_chunk(b'IDAT', compressed)

        # IEND chunk
        iend = png_chunk(b'IEND', b'')

        return signature + ihdr + idat + iend

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'wb') as f:
        f.write(create_png(size, size))

    print(f"Created placeholder icon: {output_path}")


def main():
    """Generate all required icon sizes."""
    base_path = os.path.dirname(os.path.abspath(__file__))
    icons_path = os.path.join(base_path, 'icons')

    sizes = [16, 32, 48, 128]

    print("PhishGuard Icon Generator")
    print("-" * 40)

    for size in sizes:
        output_path = os.path.join(icons_path, f'icon{size}.png')
        create_simple_icon(size, output_path)

    print("-" * 40)
    print("Icon generation complete!")
    print("\nTo load the extension:")
    print("1. Open Chrome and go to chrome://extensions/")
    print("2. Enable 'Developer mode'")
    print("3. Click 'Load unpacked'")
    print(f"4. Select the {icons_path} directory (parent folder)")


if __name__ == '__main__':
    main()