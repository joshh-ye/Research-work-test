import torch
from torch.utils.data import DataLoader

def train(model, train_dataset, val_dataset, n_epochs=3, batch_size=1, lr=1e-4):
    optimizer = torch.optim.Adam(model.head.parameters(), lr=lr)

    for epoch in range(n_epochs):
        model.head.train()

        train_loss = 0.0

        for batch in DataLoader(train_dataset, batch_size, shuffle=True):
            sequence = batch["sequence"].to(model.device)
            targets = batch["targets"].to(model.device)

            pred = model(sequence)
            
            loss = torch.nn.functional.poisson_nll_loss(pred, targets, log_input=False)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        
        avg_loss = train_loss / len(train_dataset)
        print(f"Epoch {epoch+1}/{n_epochs}  train_loss={avg_loss:.4f}")

    torch.save(model.head.state_dict(), "model_head.pt")
    print("model saved")
