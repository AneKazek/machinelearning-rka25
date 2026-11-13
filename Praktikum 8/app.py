import gradio as gr
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageOps


class CNNBN(nn.Module):
    def __init__(self, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

DEFAULT_MODEL_PATH = "model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_weights(model_path):
    model = CNNBN(dropout=0.5).to(DEVICE)
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model


default_model = load_weights(DEFAULT_MODEL_PATH)


def predict(image: Image.Image, auto_invert: bool, custom_model_file):
    if image is None:
        return None, None

    active_model = load_weights(custom_model_file) if custom_model_file is not None else default_model

    img_gray = image.convert("L")
    img_resized = img_gray.resize((28, 28), Image.Resampling.LANCZOS)

    img_array = np.array(img_resized, dtype=np.float32)
    if auto_invert and np.mean(img_array) > 127.0:
        img_resized = ImageOps.invert(img_resized)
        img_array = np.array(img_resized, dtype=np.float32)

    tensor = torch.tensor(img_array / 255.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = active_model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    confidences = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
    return confidences, img_resized


with gr.Blocks(title="Fashion-MNIST Classifier") as demo:
    gr.Markdown("# 👕 Fashion-MNIST Image Classifier\nModel **CNN + BatchNorm + Dropout** (`model.pth` sudah aktif otomatis).")

    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="pil", label="Unggah Gambar Pakaian / Alas Kaki")
            auto_invert = gr.Checkbox(value=True, label="Auto-invert latar belakang terang")
            custom_model = gr.File(label="Ganti Model (.pth) - Opsional", file_types=[".pth", ".pt"])
            btn = gr.Button("Prediksi", variant="primary")

        with gr.Column():
            output_label = gr.Label(num_top_classes=5, label="Hasil Prediksi & Probabilitas")
            preview_img = gr.Image(type="pil", label="Preview Input Model (28x28 Grayscale)", width=160, height=160)

    btn.click(fn=predict, inputs=[input_img, auto_invert, custom_model], outputs=[output_label, preview_img])

if __name__ == "__main__":
    demo.launch()