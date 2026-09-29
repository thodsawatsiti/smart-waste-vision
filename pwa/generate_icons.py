"""สร้างไอคอน PWA"""
from PIL import Image, ImageDraw, ImageFont

for size in [192, 512]:
    img = Image.new("RGB", (size, size), "#1a1a2e")
    draw = ImageDraw.Draw(img)

    # วงกลมพื้นหลัง
    margin = size // 8
    draw.ellipse(
        [margin, margin, size - margin, size - margin],
        fill="#4CAF50"
    )

    # ตัวอักษร
    font_size = size // 3
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    text = "G"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (size - text_w) // 2
    y = (size - text_h) // 2 - size // 20
    draw.text((x, y), text, fill="#fff", font=font)

    img.save(f"C:/Users/popjr/Desktop/garbage/garbage-detection-yolov8/pwa/icon-{size}.png")
    print(f"Created icon-{size}.png")

print("Done!")
