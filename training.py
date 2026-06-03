import os

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.nn.utils import clip_grad_norm_
from torch_geometric.loader import DataLoader as GeoDataLoader

from config import NUM_NODE_TOKENS, ProjectConfig
from data_processing import GraphPeptideDataset
from losses import compute_discriminator_losses, compute_generator_loss, nnpu_loss
from models import Generator, GraphCombinedDiscriminator
from self_training import SelfTrainingBuffer, generate_and_filter_samples


def default_model_config(config: ProjectConfig, full_training=False):
    return {
        "epochs": config.full_epochs if full_training else config.cv_epochs,
        "batch_size": config.batch_size,
        "num_gen_updates": config.num_generator_updates,
        "curriculum_epochs": config.curriculum_epochs,
        "gan_warmup_epochs": config.gan_warmup_epochs,
        "adv_loss_type": config.adversarial_loss_type,
    }


def create_models(config: ProjectConfig, device, pos_prior: float):
    generator = Generator(config.noise_dim, config.latent_dim).to(device)
    discriminator = GraphCombinedDiscriminator(
        NUM_NODE_TOKENS,
        config.node_emb_dim,
        config.hidden_dim,
        config.latent_dim,
        config.num_transformer_layers,
        pos_prior=pos_prior,
    ).to(device)
    return generator, discriminator


def _save_checkpoint(path, generator, discriminator, pos_prior, epoch, metrics=None, buffer_size=0, optimizers=None):
    payload = {
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "pos_prior": pos_prior,
        "epoch": epoch,
        "metrics": metrics or {},
        "self_training_buffer_size": buffer_size,
    }
    if optimizers is not None:
        payload["optimizer_G"] = optimizers[0].state_dict()
        payload["optimizer_D"] = optimizers[1].state_dict()
    torch.save(payload, path)


def train_model(
    config: ProjectConfig,
    train_dataset,
    val_dataset=None,
    output_dir="models",
    fold=None,
    class_prior_estimate=None,
    class_prior_details=None,
    model_config=None,
    device=None,
    log_interval=1,
):
    os.makedirs(output_dir, exist_ok=True)
    if model_config is None:
        model_config = default_model_config(config)

    epochs = model_config.get("epochs", config.cv_epochs)
    batch_size = model_config.get("batch_size", config.batch_size)
    num_gen_updates = model_config.get("num_gen_updates", config.num_generator_updates)
    curriculum_epochs = model_config.get("curriculum_epochs", config.curriculum_epochs)
    gan_warmup_epochs = model_config.get("gan_warmup_epochs", config.gan_warmup_epochs)
    adv_loss_type = model_config.get("adv_loss_type", config.adversarial_loss_type)

    if class_prior_estimate is None:
        label_frequency = float(np.mean(train_dataset.labels)) if len(train_dataset.labels) else 0.5
        pos_prior_dynamic = float(np.clip(label_frequency, config.class_prior_min, config.class_prior_max))
        class_prior_details = {"method": "training_label_frequency", "pi_p": pos_prior_dynamic}
    else:
        pos_prior_dynamic = float(np.clip(class_prior_estimate, config.class_prior_min, config.class_prior_max))

    generator, discriminator = create_models(config, device, pos_prior_dynamic)

    self_training_buffer = SelfTrainingBuffer(
        max_size=config.max_self_train_samples,
        latent_dim=config.latent_dim,
        min_confidence=config.min_confidence_threshold,
        min_diversity=config.diversity_threshold,
        device=device,
    )

    if val_dataset is None:
        stratify = train_dataset.labels if len(np.unique(train_dataset.labels)) > 1 else None
        train_idx, val_idx = train_test_split(
            np.arange(len(train_dataset)),
            test_size=0.1,
            random_state=config.random_state,
            stratify=stratify,
        )
        val_dataset = GraphPeptideDataset(
            [train_dataset.graphs[i] for i in val_idx],
            train_dataset.labels[val_idx],
        )
        train_dataset = GraphPeptideDataset(
            [train_dataset.graphs[i] for i in train_idx],
            train_dataset.labels[train_idx],
        )

    train_loader = GeoDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = GeoDataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer_G = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.999), weight_decay=1e-5)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999), weight_decay=1e-5)
    scheduler_G = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_G, T_0=10, T_mult=1, eta_min=1e-6)
    scheduler_D = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_D, T_0=10, T_mult=1, eta_min=1e-6)

    best_val_loss = float("inf")
    best_val_f1 = -float("inf")
    best_val_auc = -float("inf")
    no_improve = 0
    fold_str = f"fold_{fold}" if fold is not None else "full_data"
    history = {"val_adv_loss": [], "val_pu_loss": [], "val_total_loss": [], "roc": []}

    print(f"Starting training on {fold_str} with {len(train_dataset)} training samples")
    print(f"Estimated positive class prior pi_p: {pos_prior_dynamic:.4f}")
    if class_prior_details is not None:
        print(f"Class-prior estimator: {class_prior_details.get('method', 'unknown')}")

    best_path = os.path.join(output_dir, f"best_f1_model_{fold_str}.pth")

    for epoch in range(1, epochs + 1):
        generator.train()
        discriminator.train()
        epoch_adv_loss = 0.0
        epoch_pu_loss = 0.0
        epoch_gen_loss = 0.0
        count_batches = 0

        if epoch <= gan_warmup_epochs:
            use_gan = False
            curr_phase = 0.0
            gradient_reversal_alpha = 0.0
        elif epoch <= gan_warmup_epochs + curriculum_epochs:
            use_gan = True
            curr_phase = (epoch - gan_warmup_epochs) / curriculum_epochs
            gradient_reversal_alpha = 0.0
        else:
            use_gan = True
            curr_phase = 1.0
            gradient_reversal_alpha = min((epoch - gan_warmup_epochs - curriculum_epochs) * 0.05, 0.5)

        discriminator.set_gradient_reversal_alpha(gradient_reversal_alpha)
        current_fm_weight = config.feature_matching_weight * curr_phase
        current_pu_feedback_weight = config.generator_pu_feedback_weight * curr_phase

        if use_gan and epoch > gan_warmup_epochs + 5 and epoch % config.self_train_update_freq == 0:
            filtered_latents, confidence_scores = generate_and_filter_samples(
                generator,
                discriminator,
                num_samples=batch_size * 10,
                noise_dim=config.noise_dim,
                latent_dim=config.latent_dim,
                min_confidence=config.min_confidence_threshold,
                device=device,
            )
            self_training_buffer.add_samples(filtered_latents, confidence_scores)
            print(f"Epoch {epoch}: self-training buffer contains {len(self_training_buffer)} samples")

        for real_batch, label_batch in train_loader:
            real_batch = real_batch.to(device)
            label_batch = label_batch.to(device)
            batch_size_real = real_batch.num_graphs

            if use_gan and len(self_training_buffer) > 0 and epoch > gan_warmup_epochs + 5:
                gen_latents, _ = self_training_buffer.get_samples(batch_size=max(1, batch_size_real // 2))
                if gen_latents is not None and len(gen_latents) > 0:
                    optimizer_D.zero_grad()
                    gen_latents = gen_latents.to(device)
                    gen_labels = torch.ones(len(gen_latents), device=device)
                    _, real_pu_out = discriminator(real_batch, get_only_pu=True)
                    _, fake_pu_out = discriminator(gen_latents, get_only_pu=True)
                    combined_logits = torch.cat([real_pu_out.view(-1), fake_pu_out.view(-1)])
                    combined_labels = torch.cat([label_batch.view(-1), gen_labels.view(-1)])
                    d_pu_loss = nnpu_loss(combined_logits, combined_labels, pos_prior_dynamic)
                    d_total_loss = config.pu_loss_weight * d_pu_loss
                    d_total_loss.backward()
                    clip_grad_norm_(discriminator.parameters(), 1.0)
                    optimizer_D.step()

            if use_gan:
                g_loss_accum = 0.0
                for _ in range(num_gen_updates):
                    optimizer_G.zero_grad()
                    noise = torch.randn(batch_size_real, config.noise_dim, device=device)
                    g_loss, _ = compute_generator_loss(
                        generator,
                        discriminator,
                        noise,
                        real_batch,
                        feature_matching_weight=current_fm_weight,
                        pu_feedback_weight=current_pu_feedback_weight,
                        loss_type=adv_loss_type,
                    )
                    g_loss.backward()
                    clip_grad_norm_(generator.parameters(), 1.0)
                    optimizer_G.step()
                    g_loss_accum += g_loss.item()
                g_loss_avg = g_loss_accum / num_gen_updates
            else:
                g_loss_avg = 0.0

            optimizer_D.zero_grad()
            if use_gan:
                noise = torch.randn(batch_size_real, config.noise_dim, device=device)
                gen_data = generator(noise).detach()
                d_adv_loss, d_pu_loss = compute_discriminator_losses(
                    discriminator,
                    real_data=real_batch,
                    real_labels=label_batch,
                    fake_data=gen_data,
                    pi_p=pos_prior_dynamic,
                    loss_type=adv_loss_type,
                )
                d_total_loss = curr_phase * d_adv_loss + config.pu_loss_weight * d_pu_loss
            else:
                _, pu_real_out = discriminator(real_batch, get_only_pu=True)
                d_pu_loss = nnpu_loss(pu_real_out, label_batch, pos_prior_dynamic)
                d_adv_loss = torch.tensor(0.0, device=device)
                d_total_loss = config.pu_loss_weight * d_pu_loss

            d_total_loss.backward()
            clip_grad_norm_(discriminator.parameters(), 1.0)
            optimizer_D.step()

            epoch_adv_loss += d_adv_loss.item() if use_gan else 0.0
            epoch_pu_loss += d_pu_loss.item()
            epoch_gen_loss += g_loss_avg if use_gan else 0.0
            count_batches += 1

        if count_batches > 0:
            epoch_adv_loss /= count_batches
            epoch_pu_loss /= count_batches
            epoch_gen_loss /= count_batches

        generator.eval()
        discriminator.eval()
        val_adv_loss = 0.0
        val_pu_loss = 0.0
        val_labels = []
        val_probs = []
        val_batches = 0

        with torch.no_grad():
            for val_batch, val_label_batch in val_loader:
                val_batch = val_batch.to(device)
                val_label_batch = val_label_batch.to(device)
                _, pu_out_val = discriminator(val_batch, get_only_pu=True)
                d_pu_loss_val = torch.nn.functional.binary_cross_entropy_with_logits(
                    pu_out_val.view(-1), val_label_batch.view(-1).float()
                )
                probs = torch.sigmoid(pu_out_val).view(-1)
                val_pu_loss += d_pu_loss_val.item()
                val_labels.extend(val_label_batch.cpu().numpy().tolist())
                val_probs.extend(probs.cpu().numpy().tolist())
                val_batches += 1

        if val_batches > 0:
            val_pu_loss /= val_batches
        val_total = val_adv_loss + config.pu_loss_weight * val_pu_loss

        val_probs_array = np.array(val_probs).flatten()
        val_labels_array = np.array(val_labels).flatten()
        val_preds_array = (val_probs_array >= config.classification_threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            val_labels_array, val_preds_array, average="binary", zero_division=1
        )
        roc_val = roc_auc_score(val_labels_array, val_probs_array) if len(np.unique(val_labels_array)) > 1 else float("nan")
        acc_val = accuracy_score(val_labels_array, val_preds_array)

        history["val_adv_loss"].append(val_adv_loss)
        history["val_pu_loss"].append(val_pu_loss)
        history["val_total_loss"].append(val_total)
        history["roc"].append(roc_val)

        scheduler_G.step()
        scheduler_D.step()

        if epoch % log_interval == 0:
            st_count = len(self_training_buffer) if epoch > gan_warmup_epochs else 0
            print(
                f"{fold_str} | Epoch {epoch}/{epochs} | G Loss: {epoch_gen_loss:.4f} | "
                f"D Adv Loss: {epoch_adv_loss:.4f} | D PU Loss: {epoch_pu_loss:.4f} | "
                f"Val Total: {val_total:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f} | "
                f"F1: {f1:.4f} | ROC: {roc_val:.4f} | Acc: {acc_val:.4f} | "
                f"pi_p: {pos_prior_dynamic:.3f} | ST samples: {st_count}"
            )

        precision_recall_ratio = precision / (recall + 1e-8)
        metrics = {"precision": precision, "recall": recall, "f1": f1, "roc": roc_val, "accuracy": acc_val}

        if f1 > best_val_f1:
            best_val_f1 = f1
            no_improve = 0
            _save_checkpoint(
                best_path,
                generator,
                discriminator,
                pos_prior_dynamic,
                epoch,
                metrics=metrics,
                buffer_size=len(self_training_buffer),
            )
            if precision_recall_ratio > 1.2:
                pos_prior_dynamic = min(pos_prior_dynamic * 1.05, config.class_prior_max)
            elif precision_recall_ratio < 0.8:
                pos_prior_dynamic = max(pos_prior_dynamic * 0.95, config.class_prior_min)
        else:
            if precision_recall_ratio > 1.2:
                pos_prior_dynamic = min(pos_prior_dynamic * 1.1, config.class_prior_max)
            elif precision_recall_ratio < 0.8:
                pos_prior_dynamic = max(pos_prior_dynamic * 0.9, config.class_prior_min)
            no_improve += 1

        if not np.isnan(roc_val) and roc_val > best_val_auc:
            best_val_auc = roc_val
            _save_checkpoint(
                os.path.join(output_dir, f"best_auc_model_{fold_str}.pth"),
                generator,
                discriminator,
                pos_prior_dynamic,
                epoch,
                metrics=metrics,
                buffer_size=len(self_training_buffer),
            )

        if val_total < best_val_loss:
            best_val_loss = val_total
            _save_checkpoint(
                os.path.join(output_dir, f"best_loss_model_{fold_str}.pth"),
                generator,
                discriminator,
                pos_prior_dynamic,
                epoch,
                buffer_size=len(self_training_buffer),
            )

        if epoch % 10 == 0:
            _save_checkpoint(
                os.path.join(output_dir, f"checkpoint_epoch_{epoch}_{fold_str}.pth"),
                generator,
                discriminator,
                pos_prior_dynamic,
                epoch,
                metrics=metrics,
                buffer_size=len(self_training_buffer),
                optimizers=(optimizer_G, optimizer_D),
            )

        if no_improve >= 10:
            print(f"Early stopping after {epoch} epochs: no F1 improvement for 10 epochs")
            break

    print(f"Training completed for {fold_str}")
    print(f"Best F1: {best_val_f1:.4f} | Best AUC: {best_val_auc:.4f} | Best Val Loss: {best_val_loss:.4f}")

    checkpoint = torch.load(best_path, map_location=device)
    generator.load_state_dict(checkpoint["generator"])
    discriminator.load_state_dict(checkpoint["discriminator"])
    return generator, discriminator, history
