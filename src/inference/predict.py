import torch
import torchvision.transforms as transforms
from src.utils.io import load_model
from src.datasets.dataloader import DataLoader
from src.encoders.vision_encoder import VisionEncoder
from src.encoders.text_encoder import TextEncoder
from src.fusion.fusion_module import FusionModule
from src.policy.policy_transformer import PolicyTransformer

class Predictor:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model = load_model(model_path).to(self.device)
        self.vision_encoder = VisionEncoder().to(self.device)
        self.text_encoder = TextEncoder().to(self.device)
        self.fusion_module = FusionModule().to(self.device)
        self.policy_transformer = PolicyTransformer().to(self.device)

    def preprocess(self, image, text_instruction):
        image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(self.device)
        text_embedding = self.text_encoder(text_instruction).unsqueeze(0).to(self.device)
        return image_tensor, text_embedding

    def predict(self, image, text_instruction):
        image_tensor, text_embedding = self.preprocess(image, text_instruction)
        vision_embedding = self.vision_encoder(image_tensor)
        fused_representation = self.fusion_module(vision_embedding, text_embedding)
        action_logits = self.policy_transformer(fused_representation)
        return action_logits

if __name__ == "__main__":
    import argparse
    from PIL import Image

    parser = argparse.ArgumentParser(description='Inference for Visual In-Context Learning with 3D Perception')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
    parser.add_argument('--text_instruction', type=str, required=True, help='Text instruction for the model')
    args = parser.parse_args()

    image = Image.open(args.image_path).convert('RGB')
    predictor = Predictor(model_path=args.model_path)
    action_logits = predictor.predict(image, args.text_instruction)
    print("Predicted action logits:", action_logits)