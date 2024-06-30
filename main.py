from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from io import BytesIO
from PIL import Image
import torch
from torchvision import datasets, models, transforms
from numpy import asarray
# from facenet_pytorch import MTCNN, InceptionResnetV1

app = FastAPI()

app.add_middleware(CORSMiddleware, allow_origins=["*"])

# Load your YOLO model
model = YOLO("models/posm-ver2.pt")

# Load your context classifier
context_classifier = models.resnet18(weights=None, num_classes=4)
context_classifier.load_state_dict(torch.load("models/context-ver0.pt"))
preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
context_classifier.eval()
context_labels = ["Ban le", "Nha hang", "Sieu thi", "Su kien"]

brand_classifier = models.resnet18(weights=None, num_classes=7)
brand_classifier.load_state_dict(torch.load("models/brand-ver1.pt"))
brand_classifier.eval()
brand_labels = [
    "Bia Viet",
    "Bivina",
    "Edelweiss",
    "Heineken",
    "Larue",
    "Strongbow",
    "Tiger",
]

# If required, create a face detection pipeline using MTCNN:
# mtcnn = MTCNN()

# # Create an inception resnet (in eval mode):
# resnet = InceptionResnetV1(pretrained='vggface2').eval()
# resnet.classify = True

@app.get("/")
def read_root():
    return {"message": "Welcome to the YOLO API"}


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents))
    context = get_context(image)
    # Convert image to Tensor
    image = asarray(image)

    # Perform inference
    results = model.predict(image)

    # Process results
    predictions = []
    for result in results:
        summary = result.summary()
        # print(summary)
        for entry in summary:
            box = entry["box"]
            x1, y1, x2, y2 = (
                round(box["x1"]),
                round(box["y1"]),
                round(box["x2"]),
                round(box["y2"]),
            )
            cropped_image = image[y1:y2, x1:x2]
            # Apply preprocessing to the cropped image
            preprocessed_image = preprocess(Image.fromarray(cropped_image))

            # convert the image to a tensor
            input_tensor = torch.tensor(preprocessed_image)

            # Add a batch dimension
            input_batch = input_tensor.unsqueeze(0)

            # Perform inference
            output = brand_classifier(input_batch)

            _, index = torch.max(output, 1)

            brand = brand_labels[index]
            predictions.append(
                {
                    **entry,
                    "brand": brand,
                }
            )

    return {"predictions": predictions, "context": context}

def get_context(image):
    # Apply preprocessing to the PIL Image
    preprocessed_image = preprocess(image)

    # convert the image to a tensor
    input_tensor = torch.tensor(preprocessed_image)

    # Add a batch dimension
    input_batch = input_tensor.unsqueeze(0)

    # Perform inference
    output = context_classifier(input_batch)

    _, index = torch.max(output, 1)

    context = context_labels[index]

    return context

@app.post("/context/")
async def context(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents))

    # Apply preprocessing to the PIL Image
    preprocessed_image = preprocess(image)

    # convert the image to a tensor
    input_tensor = torch.tensor(preprocessed_image)

    # Add a batch dimension
    input_batch = input_tensor.unsqueeze(0)

    # Perform inference
    output = context_classifier(input_batch)

    _, index = torch.max(output, 1)

    context = context_labels[index]

    return {"context": context}

# @app.post("/face/")
# async def face(file: UploadFile = File(...)):
#     contents = await file.read()
#     image = Image.open(BytesIO(contents))

#     # Detect faces
#     faces = mtcnn(image)

#     # Calculate embeddings
#     embeddings = resnet(faces.unsqueeze(0))

#     return {"embeddings": embeddings}