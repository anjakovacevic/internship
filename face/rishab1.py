from PIL import Image, ImageDraw  # Python image library
import face_recognition

image = face_recognition.load_image_file("Traditional-Family-Life.jpg")

face_locations = face_recognition.face_locations(image)

print("I found {} face(s) in this photo.".format(len(face_locations)))

pil_image = Image.fromarray(image)
draw = ImageDraw.Draw(pil_image)

for face_location in face_locations:
    top, right, bottom, left = face_location
    print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))
    '''Izdvojiti lica sa slike
    face_image = image[top:bottom, left:right]
    pil_image = Image.fromarray(face_image)
    pil_image.show()
    '''
    draw.rectangle(((left, top), (right, bottom)), outline=(0,255,0))
pil_image.show()

