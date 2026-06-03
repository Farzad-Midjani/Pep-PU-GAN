import torch
import torch.nn as nn


def nnpu_loss(logits: torch.Tensor, labels: torch.Tensor, pi_p: float, beta: float = 0.0, gamma: float = 1.0):
    logits = logits.view(-1)
    labels = labels.view(-1)
    positive_mask = labels == 1
    unlabeled_mask = labels == 0

    pos_logits = logits[positive_mask]
    unl_logits = logits[unlabeled_mask]
    zero = logits.sum() * 0.0

    loss_pos = torch.nn.functional.softplus(-pos_logits)
    loss_neg_from_pos = torch.nn.functional.softplus(pos_logits)
    loss_neg_from_unl = torch.nn.functional.softplus(unl_logits)

    positive_risk = pi_p * loss_pos.mean() if positive_mask.sum().item() > 0 else zero
    negative_risk = (
        loss_neg_from_unl.mean() - pi_p * loss_neg_from_pos.mean()
        if unlabeled_mask.sum().item() > 0 and positive_mask.sum().item() > 0
        else zero
    )

    if negative_risk.detach().item() < -beta:
        return positive_risk - beta - gamma * (negative_risk + beta)
    return positive_risk + negative_risk


def hinge_loss_d(real_logits, fake_logits):
    return torch.mean(torch.relu(1.0 - real_logits)) + torch.mean(torch.relu(1.0 + fake_logits))


def hinge_loss_g(fake_logits):
    return -torch.mean(fake_logits)


def wasserstein_loss_d(real_logits, fake_logits):
    return -torch.mean(real_logits) + torch.mean(fake_logits)


def wasserstein_loss_g(fake_logits):
    return -torch.mean(fake_logits)


def compute_discriminator_losses(discriminator, real_data, real_labels, fake_data, pi_p, loss_type="hinge"):
    disc_real_out, pu_real_out = discriminator(real_data)
    disc_fake_out, _ = discriminator(fake_data.detach(), forward_fake=True)

    if loss_type == "wasserstein":
        adv_loss = wasserstein_loss_d(disc_real_out, disc_fake_out)
    elif loss_type == "hinge":
        adv_loss = hinge_loss_d(disc_real_out, disc_fake_out)
    else:
        loss_fn = nn.BCEWithLogitsLoss()
        real_targets = 0.9 * torch.ones_like(disc_real_out)
        fake_targets = torch.zeros_like(disc_fake_out)
        adv_loss = loss_fn(disc_real_out, real_targets) + loss_fn(disc_fake_out, fake_targets)

    pu_loss = nnpu_loss(pu_real_out, real_labels, pi_p)
    return adv_loss, pu_loss


def compute_generator_loss(
    generator,
    discriminator,
    noise,
    real_data,
    feature_matching_weight=1.0,
    pu_feedback_weight=0.5,
    loss_type="hinge",
):
    generated_data = generator(noise)
    disc_fake_out, pu_fake_out = discriminator(generated_data, forward_fake=True)

    if loss_type == "wasserstein":
        g_adv_loss = wasserstein_loss_g(disc_fake_out)
    elif loss_type == "hinge":
        g_adv_loss = hinge_loss_g(disc_fake_out)
    else:
        g_adv_loss = -torch.mean(torch.log_sigmoid(disc_fake_out))

    with torch.no_grad():
        real_features = discriminator.get_adv_features(real_data)
        real_mean = real_features.mean(dim=0)
        real_var = real_features.var(dim=0)

    fake_features = discriminator.get_adv_features(generated_data)
    fm_loss = (
        torch.nn.functional.mse_loss(fake_features.mean(dim=0), real_mean)
        + torch.nn.functional.mse_loss(fake_features.var(dim=0), real_var)
    )
    target_ones = torch.ones_like(pu_fake_out)
    gen_pu_loss = nn.BCEWithLogitsLoss()(pu_fake_out, target_ones)
    total_loss = g_adv_loss + feature_matching_weight * fm_loss + pu_feedback_weight * gen_pu_loss
    return total_loss, generated_data
