
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from appRostros import facenet_models
from PIL import Image


def process_image(image):
    face_net_models = facenet_models.FaceNetModels()
    fs = FileSystemStorage()
    filename = fs.save(image.name, image)
    path = fs.path(filename)
    img = Image.open(path)
    image_embedding = face_net_models.embedding(face_net_models.mtcnn(img))
    distancia = face_net_models.Distancia(image_embedding)
    fs.delete(filename)
    return distancia[0][0],distancia[1][0]

def upload_image(request):
    if request.method == 'POST':
        image = request.FILES['image']
        label, Distancia = process_image(image)
        return render(request, 'label.html', {'label': label, 'Distancia': Distancia})
    else:
        return render(request, 'upload.html')



