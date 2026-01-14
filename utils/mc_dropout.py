# utils/mc_dropout.py

import torch


def mc_dropout_predict(model, x, mc_samples=30):
    """
    Monte Carlo Dropout inference
    """

    model.train()  # IMPORTANT: enable dropout

    finger_probs = []
    action_probs = []

    with torch.no_grad():
        for _ in range(mc_samples):
            f_logits, a_logits = model(x)
            finger_probs.append(torch.softmax(f_logits, dim=1))
            action_probs.append(torch.softmax(a_logits, dim=1))

    model.eval()

    finger_probs = torch.stack(finger_probs)
    action_probs = torch.stack(action_probs)

    return {
        "finger_mean": finger_probs.mean(0),
        "finger_std": finger_probs.std(0),
        "action_mean": action_probs.mean(0),
        "action_std": action_probs.std(0),
    }
