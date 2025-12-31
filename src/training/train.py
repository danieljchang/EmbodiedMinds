import argparse
import os
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.datasets.dataloader import build_dataloader
from src.encoders.text_encoder import TextEncoder
from src.encoders.vision_encoder import VisionEncoder
from src.policy.policy_transformer import PolicyTransformer
from src.heads.output_heads import OutputHeads

def collate_fn(batch):
    # batch is list of dicts from EmbodiedDataset
    # We will build simple tokens: instruction embedding, demo image embeddings averaged, current image embedding
    instrs = [b["instruction"] for b in batch]
    demo_imgs = torch.stack([b["demo_images"].mean(dim=0) for b in batch], dim=0)  # (B,3,H,W)
    current_imgs = torch.stack([b["current_image"] for b in batch], dim=0)
    # targets: choose last valid action from demo_actions[0] (first demo) as training target
    targets = []
    for b in batch:
        seq = b["demo_actions"][0]  # (max_steps,7)
        # find last non -1 row
        valid = (seq != -1).all(dim=1)
        inds = valid.nonzero(as_tuple=False)
        if len(inds) == 0:
            targets.append([-1]*7)
        else:
            last = inds[-1].item()
            targets.append(seq[last].tolist())
    targets = torch.tensor(targets, dtype=torch.long)
    return instrs, demo_imgs, current_imgs, targets

def train_epoch(model_components, dataloader, optimizers, device):
    text_enc, vis_enc, policy, heads = model_components
    optimizer = optimizers
    policy.train()
    heads.train()
    pbar = tqdm(dataloader)
    total_loss = 0.0
    for instrs, demo_imgs, current_imgs, targets in pbar:
        # encode
        instr_embed = text_enc.encode(instrs)  # (B, tdim)
        demo_embed = vis_enc.encode(demo_imgs)  # (B, vdim)
        cur_embed = vis_enc.encode(current_imgs)  # (B, vdim)
        # build tokens: concat embeddings after projecting to same dim
        # simple concatenation along sequence dim
        # project to token dim
        token_dim = policy.transformer.layers[0].d_model if hasattr(policy.transformer.layers[0], "d_model") else policy.transformer.layers[0].self_attn.embed_dim
        # simple linear projections on the fly
        proj_instr = torch.nn.Linear(instr_embed.size(1), token_dim).to(device)
        proj_vis = torch.nn.Linear(demo_embed.size(1), token_dim).to(device)
        t_instr = proj_instr(instr_embed)
        t_demo = proj_vis(demo_embed)
        t_cur = proj_vis(cur_embed)
        # tokens as sequence length 3: instr, demo, cur
        tokens = torch.stack([t_instr, t_demo, t_cur], dim=1)  # (B, 3, token_dim)
        decision = policy(tokens)  # (B, out_dim)
        logits = heads(decision)
        loss = heads.loss(logits, targets.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_description(f"loss {total_loss / (1 + pbar.n):.4f}")
    return total_loss / len(dataloader)

def eval_epoch(model_components, dataloader, device):
    text_enc, vis_enc, policy, heads = model_components
    policy.eval(); heads.eval()
    correct = [0]*7
    total = 0
    with torch.no_grad():
        for instrs, demo_imgs, current_imgs, targets in tqdm(dataloader):
            instr_embed = text_enc.encode(instrs)
            demo_embed = vis_enc.encode(demo_imgs)
            cur_embed = vis_enc.encode(current_imgs)
            token_dim = policy.transformer.layers[0].d_model if hasattr(policy.transformer.layers[0], "d_model") else policy.transformer.layers[0].self_attn.embed_dim
            proj_instr = torch.nn.Linear(instr_embed.size(1), token_dim).to(device)
            proj_vis = torch.nn.Linear(demo_embed.size(1), token_dim).to(device)
            t_instr = proj_instr(instr_embed)
            t_demo = proj_vis(demo_embed)
            t_cur = proj_vis(cur_embed)
            tokens = torch.stack([t_instr, t_demo, t_cur], dim=1)
            decision = policy(tokens)
            logits = heads(decision)
            preds = heads.predict([l.to(device) for l in logits]).cpu()
            tgt = targets
            mask = (tgt != -1)
            for i in range(7):
                valid = mask[:, i]
                if valid.sum().item() == 0:
                    continue
                correct[i] += (preds[valid, i] == tgt[valid, i]).sum().item()
            total += tgt.size(0)
    accs = [c / max(1, total) for c in correct]
    return accs

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=False, help="Path to config (optional)")
    p.add_argument("--debug", action="store_true", help="Run quick debug mode")
    p.add_argument("--data-root", required=False, help="Path to data folder (overrides EMBODIEDBENCH_DATA)")
    return p.parse_args()

def train_model(config):
    # Load dataset
    train_dataset = CustomDataset(config['data']['train'])
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)

    # Initialize models
    vision_encoder = VisionEncoder(config['model']['vision'])
    text_encoder = TextEncoder(config['model']['text'])
    fusion_module = FusionModule(config['model']['fusion'])
    policy_transformer = PolicyTransformer(config['model']['policy'])

    # Set up optimizer
    optimizer = optim.Adam(policy_transformer.parameters(), lr=config['training']['learning_rate'])

    # Training loop
    for epoch in range(config['training']['epochs']):
        policy_transformer.train()
        total_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()
            vision_data = vision_encoder(batch['images'])
            text_data = text_encoder(batch['texts'])
            fused_data = fusion_module(vision_data, text_data)
            outputs = policy_transformer(fused_data)

            loss = calculate_loss(outputs, batch['labels'])
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{config["training"]["epochs"]}], Loss: {total_loss / len(train_loader):.4f}')

    # Save the trained model
    save_model(policy_transformer, config['training']['model_save_path'])

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Determine data root: CLI > ENV > fallbacks inside dataset
    data_root = args.data_root or os.environ.get("EMBODIEDBENCH_DATA", None)

    dl = build_dataloader(batch_size=4, debug=args.debug, data_root=data_root)

    # Minimal verification pass
    for batch in dl:
        print("Loaded batch. sample meta:", batch.get("meta_path", None))
        print("image shape:", batch["image"].shape)
        print("objects shape:", batch["objects"].shape)
        print("action_labels shape:", batch["action_labels"].shape)
        break

if __name__ == "__main__":
    with open(os.path.join('configs', 'train.yaml'), 'r') as file:
        config = yaml.safe_load(file)
    train_model(config)
    main()