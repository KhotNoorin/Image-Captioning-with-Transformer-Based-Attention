import streamlit as st
st.set_page_config(page_title="Image Caption Generator", layout="centered")

from PIL import Image
import torch
import yaml
import torchvision.transforms as transforms

from models.image_captioning import ImageCaptioningModel
from utils.tokenizer import Tokenizer

# Load config
with open("config/config.yaml", 'r') as f:
    config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer
tokenizer = Tokenizer()
tokenizer.load_vocab(config['processed_data']['vocab_path'])

# Load model
@st.cache_resource
def load_model():
    model = ImageCaptioningModel(
        vocab_size=len(tokenizer.word2idx),
        embed_size=config['model']['embed_size'],
        decoder_dim=config['model']['decoder_dim'],
        attention_dim=config['model']['attention_dim'],
        dropout=config['model']['dropout'],
        max_len=config['model']['max_len']
    )
    model.load_state_dict(torch.load(config['train']['save_dir'] + "best_model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def generate_caption(image):
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        memory = model.encoder(image_tensor)
        caption_idxs = [tokenizer.word2idx["<START>"]]

        for _ in range(config['model']['max_len']):
            caption_tensor = torch.tensor(caption_idxs, dtype=torch.long).unsqueeze(0).to(device)
            tgt_mask = model.generate_square_subsequent_mask(caption_tensor.size(1)).to(device)
            output = model.decoder(caption_tensor, memory, tgt_mask=tgt_mask)
            next_token = output[:, -1, :].argmax(dim=-1).item()
            caption_idxs.append(next_token)
            if next_token == tokenizer.word2idx["<END>"]:
                break

        caption = tokenizer.decode(caption_idxs[1:-1])
        return caption

# Streamlit UI
st.markdown("<h1 style='text-align: center; color: #4a4a4a;'>🖼️ Image Captioning with Transformers</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Upload an image and get a description in natural language!</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=100)  
    
    with st.spinner("Generating caption..."):
        caption = generate_caption(image)
    
    st.markdown(f"<h3 style='color: #2c3e50;'>Caption:</h3><p style='font-size: 20px;'>{caption}</p>", unsafe_allow_html=True)
