from PIL import Image

# 打开并加载图片
images = []
for i in range(1, 10):
    image = Image.open(f"{i}.png")  # 替换为你的图片路径和格式
    images.append(image)

# 获取单张图片的宽度和高度
image_width, image_height = images[0].size

# 创建新的组合图片
combined_width = image_width * 3
combined_height = image_height * 3
combined_image = Image.new('RGB', (combined_width, combined_height))

# 拼接图片
for i in range(3):
    for j in range(3):
        image_index = i * 3 + j
        combined_image.paste(images[image_index], (j * image_width, i * image_height))

# 保存组合图片
combined_image.save("combined_image.jpg")  # 替换为你想保存的文件路径和格式



