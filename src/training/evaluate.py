import torch
from src.datasets.dataloader import EmbodiedDataset
from src.encoders.text_encoder import TextEncoder
from src.encoders.vision_encoder import VisionEncoder
from src.policy.policy_transformer import PolicyTransformer
from src.heads.output_heads import OutputHeads

def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_accuracy = 0
    num_batches = len(dataloader)

    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data['input'].to(device), data['label'].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total_accuracy += (predicted == labels).sum().item()

    average_loss = total_loss / num_batches
    average_accuracy = total_accuracy / len(dataloader.dataset)

    return average_loss, average_accuracy

def predict_single(checkpoint_path, data_root, idx=0, device="cpu"):
    ds = EmbodiedDataset(data_root=data_root, debug=True)
    item = ds[idx]
    text_enc = TextEncoder(device=device)
    vis_enc = VisionEncoder(device=device)
    token_dim = 256
    out_dim = 512
    policy = PolicyTransformer(token_dim=token_dim, out_dim=out_dim).to(device)
    bins = [101,101,101,121,121,121,2]
    heads = OutputHeads(in_dim=out_dim, bins=bins).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    policy.load_state_dict(ckpt["policy"])
    heads.load_state_dict(ckpt["heads"])
    policy.eval(); heads.eval()
    instr = [item["instruction"]]
    demo_img = item["demo_images"].mean(dim=0).unsqueeze(0)
    cur_img = item["current_image"].unsqueeze(0)
    with torch.no_grad():
        ie = text_enc.encode(instr)
        di = vis_enc.encode(demo_img)
        ci = vis_enc.encode(cur_img)
        proj_instr = torch.nn.Linear(ie.size(1), token_dim).to(device)
        proj_vis = torch.nn.Linear(di.size(1), token_dim).to(device)
        t_instr = proj_instr(ie)
        t_demo = proj_vis(di)
        t_cur = proj_vis(ci)
        tokens = torch.stack([t_instr, t_demo, t_cur], dim=1)
        dec = policy(tokens)
        logits = heads(dec)
        preds = heads.predict(logits)
    return preds[0].tolist()

def main():
    # Load model and data
    model = load_model()  # Implement this function to load your model
    dataloader = load_dataloader()  # Implement this function to load your dataloader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Evaluate the model
    loss, accuracy = evaluate_model(model, dataloader, device)
    print(f"Evaluation Loss: {loss:.4f}, Evaluation Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
    print(predict_single("checkpoints/ckpt_epoch0.pt", data_root="YOUR_DATA_ROOT", idx=0, device="cpu"))