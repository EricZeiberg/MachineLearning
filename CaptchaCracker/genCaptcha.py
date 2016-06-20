from captcha.image import ImageCaptcha
import string
import random

path = "images/"

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

image = ImageCaptcha(fonts=['../OpenSans-Regular.ttf'], width=250)

for x in range(0, 500):
    randString = id_generator()
    image.write(randString, path + randString + "-" + str(x) + ".png")
