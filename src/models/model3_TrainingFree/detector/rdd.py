"""AeroBlade 재구성 오차와 RDD 디바이어싱 점수.

AeroBlade: 잠재 확산 모델의 오토인코더가 AI 이미지를 실제 사진보다 더 잘 복원한다.
RDD: 그 오차를 저역통과·90도 회전 영상 오차로 빼서, 흰 배경처럼 쉬운 사진의 편향을 줄인다.
잠재 점수는 확산 UNet이 원본 잠재벡터와 회전된 잠재벡터의 노이즈를 얼마나 잘 맞추는지의 차이다.
"""

from __future__ import annotations

import zlib
from pathlib import Path

import torch
from torchvision.transforms.functional import gaussian_blur

from detector.images import prepare_vae_input
from detector.scores import rdd_score, simage

VAE_REPOS = {
    "sd15": "runwayml/stable-diffusion-v1-5",
    "sd14": "CompVis/stable-diffusion-v1-4",
    "sd2": "stabilityai/stable-diffusion-2-base",
}


def resolve_repo(name: str) -> str:
    return VAE_REPOS.get(name, name)


def _empty_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class LayerLPIPS:
    """AeroBlade 기본 거리인 VGG LPIPS의 두 번째 층.

    공식 코드의 lpips_vgg_2는 합산 점수(인덱스 0) 다음의 두 번째 층이다.
    lpips 패키지의 층 인덱스로는 1이다.
    """

    def __init__(self, device: torch.device, layer_index: int = 1) -> None:
        import lpips

        self.layer_index = layer_index
        self.device = device
        self.model = lpips.LPIPS(net="vgg", verbose=False).to(device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.inference_mode()
    def __call__(self, original: torch.Tensor, reconstruction: torch.Tensor) -> torch.Tensor:
        # LPIPS normalize=True 는 [0, 1] 입력을 [-1, 1]로 바꾼다.
        original_01 = (original.clamp(-1, 1) + 1.0) * 0.5
        recon_01 = (reconstruction.clamp(-1, 1) + 1.0) * 0.5
        _total, layers = self.model(original_01, recon_01, retPerLayer=True, normalize=True)
        distance = layers[self.layer_index]
        return distance.reshape(distance.shape[0], -1).mean(dim=1)


def _load_subfolder(cls, repo: str, subfolder: str | None):
    kwargs = {}
    if subfolder is not None:
        kwargs["subfolder"] = subfolder
    try:
        return cls.from_pretrained(repo, **kwargs)
    except Exception:
        if subfolder is None:
            raise
        return cls.from_pretrained(repo)


def load_vae(name: str, device: torch.device):
    from diffusers import AutoencoderKL

    repo = resolve_repo(name)
    try:
        vae = _load_subfolder(AutoencoderKL, repo, "vae")
    except Exception as error:
        raise RuntimeError(
            "Stable Diffusion 오토인코더를 불러오지 못했습니다. "
            "처음 실행이면 인터넷으로 가중치를 받습니다.\n"
            f"요청한 모델: {name} ({repo})\n{error}"
        ) from error
    vae.to(device)
    vae.eval()
    vae.requires_grad_(False)
    return vae


def reconstruction_views(image: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    """원본, 저역통과, 90도 회전, 회전 후 저역통과. 배치 차원으로 쌓는다."""
    if kernel_size % 2 == 0:
        raise ValueError("가우시안 커널 크기는 홀수여야 합니다.")
    blurred = gaussian_blur(image, [kernel_size, kernel_size], [sigma, sigma])
    rotated = torch.rot90(image, 1, [-2, -1])
    blurred_rotated = gaussian_blur(rotated, [kernel_size, kernel_size], [sigma, sigma])
    return torch.cat([image, blurred, rotated, blurred_rotated], dim=0)


class ImageDebiaser:
    def __init__(self, vae, distance: LayerLPIPS, crop_size: int, kernel_size: int, sigma: float, lambda_r: float):
        self.vae = vae
        self.distance = distance
        self.crop_size = crop_size
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.lambda_r = lambda_r
        self.device = next(vae.parameters()).device

    @torch.inference_mode()
    def score_tensor(self, image: torch.Tensor) -> dict[str, float]:
        views = reconstruction_views(image.to(self.device), self.kernel_size, self.sigma)
        latent = self.vae.encode(views).latent_dist.mode()
        reconstruction = self.vae.decode(latent).sample.clamp(-1, 1)
        distances = self.distance(views, reconstruction)
        f_x, f_low, f_rot, f_both = (float(distances[i].item()) for i in range(4))
        return {
            "f_ae": f_x,
            "s_image": simage(f_x, f_low, f_rot, f_both, self.lambda_r),
        }

    def score_path(self, path: str | Path) -> dict[str, float]:
        tensor, _size = prepare_vae_input(path, self.crop_size)
        return self.score_tensor(tensor)


def _null_embedding(tokenizer, text_encoder, device: torch.device) -> torch.Tensor:
    tokens = tokenizer(
        "",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.inference_mode():
        embedding = text_encoder(tokens.input_ids.to(device))[0]
    return embedding


def _timestep_index(scheduler, timestep_fraction: float) -> int:
    count = int(scheduler.config.num_train_timesteps)
    index = int(round(float(timestep_fraction) * (count - 1)))
    return int(min(max(index, 1), count - 1))


class LatentDebiaser:
    """S_latent = f(R(z)) - f(z). 값이 클수록 실제 사진 쪽이다."""

    def __init__(self, name: str, device: torch.device, timestep: float, noise_samples: int, crop_size: int):
        from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
        from transformers import CLIPTextModel, CLIPTokenizer

        repo = resolve_repo(name)
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        try:
            self.vae = _load_subfolder(AutoencoderKL, repo, "vae")
            self.unet = _load_subfolder(UNet2DConditionModel, repo, "unet")
            self.text_encoder = _load_subfolder(CLIPTextModel, repo, "text_encoder")
            self.tokenizer = _load_subfolder(CLIPTokenizer, repo, "tokenizer")
            self.scheduler = _load_subfolder(DDPMScheduler, repo, "scheduler")
        except Exception as error:
            raise RuntimeError(
                "확산 UNet을 불러오지 못했습니다. 잠재 점수는 `--latent`를 켰을 때만 필요합니다.\n"
                f"요청한 모델: {name} ({repo})\n{error}"
            ) from error

        self.device = device
        self.timestep = timestep
        self.noise_samples = noise_samples
        self.crop_size = crop_size
        self.vae.to(device=device, dtype=torch.float32).eval()
        self.unet.to(device=device, dtype=dtype).eval()
        self.text_encoder.to(device=device, dtype=dtype).eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.null_embedding = _null_embedding(self.tokenizer, self.text_encoder, device)
        self.t_index = _timestep_index(self.scheduler, timestep)
        alpha = self.scheduler.alphas_cumprod[self.t_index]
        self.alpha = alpha.to(device=device, dtype=torch.float32)

    def close(self) -> None:
        del self.vae, self.unet, self.text_encoder, self.null_embedding
        _empty_cuda()

    @torch.inference_mode()
    def _denoising_mse(self, latent: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.to(dtype=latent.dtype)
        noisy = alpha.sqrt() * latent + (1.0 - alpha).sqrt() * noise
        timestep = torch.tensor([self.t_index], device=latent.device, dtype=torch.long)
        unet_dtype = next(self.unet.parameters()).dtype
        prediction = self.unet(
            noisy.to(unet_dtype),
            timestep,
            encoder_hidden_states=self.null_embedding.to(unet_dtype),
        ).sample.float()
        return torch.mean((prediction - noise.float()) ** 2)

    @torch.inference_mode()
    def score_path(self, path: str | Path) -> float:
        tensor, _size = prepare_vae_input(path, self.crop_size)
        tensor = tensor.to(self.device)
        latent = self.vae.encode(tensor).latent_dist.mode() * self.vae.config.scaling_factor
        rotated = torch.rot90(latent, 1, [-2, -1])
        seed = zlib.crc32(str(path).encode("utf-8")) & 0xFFFFFFFF
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        original_errors = []
        rotated_errors = []
        for _ in range(self.noise_samples):
            noise = torch.randn(latent.shape, generator=generator, dtype=torch.float32).to(self.device)
            original_errors.append(self._denoising_mse(latent, noise))
            rotated_errors.append(self._denoising_mse(rotated, noise))
        original = torch.stack(original_errors).mean()
        rotated_error = torch.stack(rotated_errors).mean()
        return float((rotated_error - original).item())


def combine_image_scores(per_vae: list[dict[str, float]], s_latent_value: float | None) -> dict[str, float]:
    f_values = [item["f_ae"] for item in per_vae]
    s_values = [item["s_image"] for item in per_vae]
    s_image_value = float(sum(s_values) / len(s_values))
    result = {
        "f_ae": float(min(f_values)),
        "f_ae_mean": float(sum(f_values) / len(f_values)),
        "s_image": s_image_value,
    }
    if s_latent_value is not None:
        result["s_latent"] = float(s_latent_value)
        result["s_rdd"] = float(rdd_score(s_image_value, s_latent_value))
    return result
