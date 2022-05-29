from PIL import Image  # using pillow-simd for increased speed


with open("index.jpg", 'rb') as f:
        with Image.open(f) as img:
            print(type(img))
            img = img.resize((1590,480))
            img = img.crop((475,0,475+640,480))
            img.show()
            img = img.convert('RGB')
            print(type(img))
