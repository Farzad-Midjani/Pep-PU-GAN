from collections import deque

import torch


class SelfTrainingBuffer:
    def __init__(self, max_size, latent_dim, min_confidence, min_diversity, device):
        self.max_size = max_size
        self.latent_dim = latent_dim
        self.min_confidence = min_confidence
        self.min_diversity = min_diversity
        self.device = device
        self.samples = deque(maxlen=max_size)
        self.feature_mean = None
        self.feature_std = None
        self.count = 0

    def update_statistics(self, features):
        features = features.detach().to(self.device)
        batch_mean = features.mean(dim=0)
        batch_std = features.std(dim=0).clamp_min(1e-8)

        if self.feature_mean is None:
            self.feature_mean = batch_mean
            self.feature_std = batch_std
        else:
            momentum = 0.95
            self.feature_mean = momentum * self.feature_mean + (1.0 - momentum) * batch_mean
            self.feature_std = momentum * self.feature_std + (1.0 - momentum) * batch_std
        self.count += 1

    def is_diverse_sample(self, latent_vector):
        if not self.samples:
            return True
        latent_vector = latent_vector.to(self.device)
        existing_vectors = torch.stack([sample[0] for sample in self.samples])
        distances = torch.norm(existing_vectors - latent_vector.unsqueeze(0), dim=1)
        return torch.min(distances).item() > self.min_diversity

    def is_within_distribution(self, latent_vector, std_threshold=3.0):
        if self.feature_mean is None or self.count < 10:
            return True
        latent_vector = latent_vector.to(self.device)
        z_scores = torch.abs((latent_vector - self.feature_mean) / (self.feature_std + 1e-8))
        return torch.mean(z_scores).item() < std_threshold

    def add_samples(self, latent_vectors, confidence_scores):
        if len(latent_vectors) == 0:
            return
        self.update_statistics(latent_vectors)

        for latent, confidence in zip(latent_vectors, confidence_scores):
            if confidence.item() < self.min_confidence:
                continue
            if not self.is_diverse_sample(latent) or not self.is_within_distribution(latent):
                continue
            self.samples.append((latent.detach().to(self.device), float(confidence.item())))

    def get_samples(self, batch_size=None):
        if not self.samples:
            return None, None

        if batch_size is None or batch_size >= len(self.samples):
            selected = list(self.samples)
        else:
            indices = torch.randperm(len(self.samples))[:batch_size]
            selected = [self.samples[int(i)] for i in indices]

        latent_tensor = torch.stack([item[0] for item in selected])
        confidence_tensor = torch.tensor([item[1] for item in selected], device=self.device)
        return latent_tensor, confidence_tensor

    def __len__(self):
        return len(self.samples)


def generate_and_filter_samples(
    generator,
    discriminator,
    num_samples,
    noise_dim,
    latent_dim,
    batch_size=64,
    min_confidence=0.75,
    device=None,
):
    generator.eval()
    discriminator.eval()
    filtered_latents = []
    confidence_scores = []

    with torch.no_grad():
        remaining = num_samples
        while remaining > 0:
            current_batch = min(batch_size, remaining)
            remaining -= current_batch
            noise = torch.randn(current_batch, noise_dim, device=device)
            fake_latents = generator(noise)
            _, pu_fake_out = discriminator(fake_latents, get_only_pu=True)
            probs = torch.sigmoid(pu_fake_out).view(-1)
            high_conf_mask = probs >= min_confidence

            if high_conf_mask.sum() > 0:
                filtered_latents.append(fake_latents[high_conf_mask])
                confidence_scores.append(probs[high_conf_mask])

    if filtered_latents:
        return torch.cat(filtered_latents, dim=0), torch.cat(confidence_scores, dim=0)
    return torch.empty((0, latent_dim), device=device), torch.empty(0, device=device)
