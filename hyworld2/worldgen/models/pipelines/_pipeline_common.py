# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import html
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import PIL
import regex as re
import torch
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput
from diffusers.utils import is_ftfy_available, is_torch_xla_available, logging
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor

try:
    from ...src.vae_utils import keyframe_vae_encode, keyframe_vae_decode
except ImportError:
    from src.vae_utils import keyframe_vae_encode, keyframe_vae_decode

if is_ftfy_available():
    import ftfy

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class PipelineCallConfig:
    height: int = 480
    width: int = 768
    num_frames: int = 81
    num_inference_steps: int = 40
    guidance_scale: float = 5.0
    num_videos_per_prompt: int = 1
    output_type: str = "np"
    return_dict: bool = True
    max_sequence_length: int = 512
    latent_cond_mode: str = "full_vae"


@dataclass
class ConditionInputs:
    image: PipelineImageInput
    render_video: torch.Tensor
    render_mask: torch.Tensor
    camera_embedding: Optional[torch.Tensor] = None
    extrinsics: Optional[torch.Tensor] = None
    intrinsics: Optional[torch.Tensor] = None
    reference_video: Optional[torch.Tensor] = None
    ref_index: Any = None
    camera_qt: Optional[torch.Tensor] = None
    camera_qt_ref: Optional[torch.Tensor] = None


@dataclass
class PreparedBatch:
    batch_size: int
    device: torch.device
    transformer_dtype: torch.dtype
    prompt_embeds: torch.Tensor
    negative_prompt_embeds: Optional[torch.Tensor]
    image_embeds: torch.Tensor
    timesteps: Any
    latents: torch.Tensor
    condition: torch.Tensor
    render_latent: torch.Tensor
    reference_latent: Optional[torch.Tensor] = None


@dataclass
class DenoiseStepInputs:
    latent_model_input: torch.Tensor
    timestep: torch.Tensor
    prompt_embeds: torch.Tensor
    negative_prompt_embeds: Optional[torch.Tensor]
    image_embeds: torch.Tensor
    render_latent: torch.Tensor
    reference_latent: Optional[torch.Tensor]


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def prompt_clean(text):
    text = whitespace_clean(basic_clean(text))
    return text


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    if hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    if hasattr(encoder_output, "latents"):
        return encoder_output.latents
    raise AttributeError("Could not access latents of provided encoder_output")


class KeyframePipelineMixin:
    def _init_keyframe_pipeline_modules(
        self,
        *,
        tokenizer,
        text_encoder,
        image_encoder,
        image_processor,
        transformer,
        vae,
        scheduler,
    ):
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            image_encoder=image_encoder,
            transformer=transformer,
            scheduler=scheduler,
            image_processor=image_processor,
        )

        self.vae_scale_factor_temporal = 2 ** sum(self.vae.temperal_downsample) if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = 2 ** len(self.vae.temperal_downsample) if getattr(self, "vae", None) else 8
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
        self.image_processor = image_processor

    def _pipeline_execution_device(self):
        execution_device = self._execution_device
        return execution_device() if callable(execution_device) else execution_device

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._pipeline_execution_device()
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
        )

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds

    def encode_image(
        self,
        image: PipelineImageInput,
        device: Optional[torch.device] = None,
    ):
        device = device or self._pipeline_execution_device()
        image = self.image_processor(images=image, return_tensors="pt").to(device)
        image_embeds = self.image_encoder(**image, output_hidden_states=True)
        return image_embeds.hidden_states[-2]

    # Copied from diffusers.pipelines.wan.pipeline_wan.WanPipeline.encode_prompt
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._pipeline_execution_device()

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            if batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, negative_prompt_embeds

    def check_inputs(
        self,
        prompt,
        negative_prompt,
        image,
        height,
        width,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        image_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if image is not None and image_embeds is not None:
            raise ValueError(
                f"Cannot forward both `image`: {image} and `image_embeds`: {image_embeds}. Please make sure to"
                " only forward one of the two."
            )
        if image is None and image_embeds is None:
            raise ValueError(
                "Provide either `image` or `prompt_embeds`. Cannot leave both `image` and `image_embeds` undefined."
            )
        if image is not None and not isinstance(image, torch.Tensor) and not isinstance(image, PIL.Image.Image):
            raise ValueError(f"`image` has to be of type `torch.Tensor` or `PIL.Image.Image` but is {type(image)}")
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 16 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`: {negative_prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        if prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        if prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        if negative_prompt is not None and (not isinstance(negative_prompt, str) and not isinstance(negative_prompt, list)):
            raise ValueError(f"`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}")

    def prepare_latents(
        self,
        image: PipelineImageInput,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        latent_cond_mode="full_vae",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial

        shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, latent_width)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        image = image.unsqueeze(2)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )

        if latent_cond_mode == "full_vae":
            video_condition = torch.cat(
                [image, image.new_zeros(image.shape[0], image.shape[1], num_frames - 1, height, width)], dim=2
            )
            video_condition = video_condition.to(device=device, dtype=self.vae.dtype)

            if isinstance(generator, list):
                latent_condition = [
                    retrieve_latents(self.vae.encode(video_condition), sample_mode="argmax") for _ in generator
                ]
                latent_condition = torch.cat(latent_condition)
            else:
                latent_condition = retrieve_latents(self.vae.encode(video_condition), sample_mode="argmax")
                latent_condition = latent_condition.repeat(batch_size, 1, 1, 1, 1)

            latent_condition = (latent_condition - latents_mean) * latents_std
        elif latent_cond_mode == "first_frame_only":
            first_frame = image.to(device=device, dtype=self.vae.dtype)
            first_frame_latent = retrieve_latents(self.vae.encode(first_frame), sample_mode="argmax")
            first_frame_latent = first_frame_latent.repeat(batch_size, 1, 1, 1, 1)
            first_frame_latent = (first_frame_latent - latents_mean) * latents_std

            zero_image = image.new_zeros(image.shape[0], image.shape[1], 1, height, width).to(
                device=device, dtype=self.vae.dtype
            )
            zero_latents = retrieve_latents(self.vae.encode(zero_image), sample_mode="argmax")
            zero_latents = zero_latents.repeat(batch_size, 1, num_latent_frames - 1, 1, 1)
            zero_latents = (zero_latents - latents_mean) * latents_std
            latent_condition = torch.cat([first_frame_latent, zero_latents], dim=2)
        else:
            raise NotImplementedError(f"latent_cond_mode {latent_cond_mode} not implemented")

        mask_lat_size = torch.zeros(
            batch_size,
            self.vae_scale_factor_temporal,
            num_latent_frames,
            latent_height,
            latent_width,
            device=latent_condition.device,
            dtype=latent_condition.dtype,
        )
        mask_lat_size[:, :, 0, :, :] = 1.0

        return latents, torch.concat([mask_lat_size, latent_condition], dim=1)

    def _normalize_callback_inputs(self, callback_on_step_end, callback_on_step_end_tensor_inputs):
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            return callback_on_step_end.tensor_inputs
        return callback_on_step_end_tensor_inputs

    def _validate_and_init_call(
        self,
        *,
        prompt,
        negative_prompt,
        condition_inputs: ConditionInputs,
        config: PipelineCallConfig,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        image_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        guidance_scale: Optional[float] = None,
    ):
        self.check_inputs(
            prompt,
            negative_prompt,
            condition_inputs.image,
            config.height,
            config.width,
            prompt_embeds,
            negative_prompt_embeds,
            image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        if config.num_frames % self.vae_scale_factor_temporal != 1:
            logger.warning(
                f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the nearest number."
            )
            config.num_frames = (
                config.num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
            )
        config.num_frames = max(config.num_frames, 1)

        self._guidance_scale = config.guidance_scale if guidance_scale is None else guidance_scale
        self._attention_kwargs = None
        self._current_timestep = None
        self._interrupt = False
        return config

    def _get_batch_size(self, prompt, prompt_embeds):
        if prompt is not None and isinstance(prompt, str):
            return 1
        if prompt is not None and isinstance(prompt, list):
            return len(prompt)
        return prompt_embeds.shape[0]

    def _prepare_prompt_and_image(
        self,
        *,
        prompt,
        negative_prompt,
        prompt_embeds,
        negative_prompt_embeds,
        image_embeds,
        image,
        config: PipelineCallConfig,
        device,
        do_classifier_free_guidance: bool,
        no_grad: bool = False,
    ):
        batch_size = self._get_batch_size(prompt, prompt_embeds)
        transformer_dtype = self.transformer.dtype

        if no_grad:
            with torch.no_grad():
                prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    num_videos_per_prompt=config.num_videos_per_prompt,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    max_sequence_length=config.max_sequence_length,
                    device=device,
                )
        else:
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=do_classifier_free_guidance,
                num_videos_per_prompt=config.num_videos_per_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                max_sequence_length=config.max_sequence_length,
                device=device,
            )

        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        if image_embeds is None:
            if no_grad:
                with torch.no_grad():
                    image_embeds = self.encode_image(image, device)
            else:
                image_embeds = self.encode_image(image, device)
        image_embeds = image_embeds.repeat(batch_size, 1, 1)
        image_embeds = image_embeds.to(transformer_dtype)

        return batch_size, transformer_dtype, prompt_embeds, negative_prompt_embeds, image_embeds

    def _prepare_standard_timesteps(self, config: PipelineCallConfig, device):
        self.scheduler.set_timesteps(config.num_inference_steps, device=device)
        return self.scheduler.timesteps

    def _prepare_dmd_timesteps(self, mode: str):
        if mode == "train":
            return self.scheduler.gen_train_timesteps()
        return self.scheduler.gen_test_timesteps()

    def _prepare_base_latents(
        self,
        *,
        image,
        batch_size: int,
        config: PipelineCallConfig,
        device,
        generator,
        latents,
    ):
        num_channels_latents = self.vae.config.z_dim
        image = self.video_processor.preprocess(image, height=config.height, width=config.width).to(
            device, dtype=torch.float32
        )
        return self.prepare_latents(
            image,
            batch_size * config.num_videos_per_prompt,
            num_channels_latents,
            config.height,
            config.width,
            config.num_frames,
            torch.float32,
            device,
            generator,
            latents,
            config.latent_cond_mode,
        )

    def _normalize_vae_latents(self, latents, ref_tensor):
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(ref_tensor.device, ref_tensor.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            ref_tensor.device, ref_tensor.dtype
        )
        return (latents - latents_mean) * latents_std

    def _prepare_render_latent(
        self,
        render_video,
        *,
        config: PipelineCallConfig,
        ref_tensor,
        use_compile: bool = False,
        compile_mode: str = "max-autotune",
        force_keyframe_encode: bool = False,
    ):
        render_video = render_video.to(dtype=self.vae.dtype)
        if force_keyframe_encode or render_video.shape[2] == config.num_frames // 4 + 1:
            if use_compile:
                return keyframe_vae_encode(
                    self.vae, render_video, rescale=True, use_compile=True, compile_mode=compile_mode
                )
            return keyframe_vae_encode(self.vae, render_video, rescale=True)

        render_latent = retrieve_latents(self.vae.encode(render_video), sample_mode="argmax")
        ref_tensor = render_latent if ref_tensor is None else ref_tensor
        return self._normalize_vae_latents(render_latent, ref_tensor)

    def _prepare_reference_latent(
        self,
        reference_video,
        *,
        use_compile: bool = False,
        compile_mode: str = "max-autotune",
        cast_to_vae_dtype: bool = True,
    ):
        if reference_video is None:
            return None

        if cast_to_vae_dtype:
            reference_video = reference_video.to(dtype=self.vae.dtype)
        if use_compile:
            reference_latent = keyframe_vae_encode(
                self.vae, reference_video, rescale=True, use_compile=True, compile_mode=compile_mode
            )
        else:
            reference_latent = keyframe_vae_encode(self.vae, reference_video, rescale=True)
        reference_latent_mask = torch.ones_like(reference_latent[:, :4]).to(
            reference_latent.dtype
        ).to(reference_latent.device)
        return torch.cat([reference_latent, reference_latent_mask, reference_latent], dim=1)

    def _prepare_standard_batch(
        self,
        *,
        prompt,
        negative_prompt,
        prompt_embeds,
        negative_prompt_embeds,
        image_embeds,
        condition_inputs: ConditionInputs,
        config: PipelineCallConfig,
        generator,
        latents,
        device,
        timesteps,
        do_classifier_free_guidance: bool,
        no_grad_prompt: bool = False,
        use_compile: bool = False,
        compile_mode: str = "max-autotune",
        latent_autocast: bool = False,
        force_keyframe_render_latent: bool = False,
        cast_reference_to_vae_dtype: bool = True,
    ) -> PreparedBatch:
        batch_size, transformer_dtype, prompt_embeds, negative_prompt_embeds, image_embeds = (
            self._prepare_prompt_and_image(
                prompt=prompt,
                negative_prompt=negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                image_embeds=image_embeds,
                image=condition_inputs.image,
                config=config,
                device=device,
                do_classifier_free_guidance=do_classifier_free_guidance,
                no_grad=no_grad_prompt,
            )
        )

        if latent_autocast:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                latents, condition = self._prepare_base_latents(
                    image=condition_inputs.image,
                    batch_size=batch_size,
                    config=config,
                    device=device,
                    generator=generator,
                    latents=latents,
                )
                render_latent = self._prepare_render_latent(
                    condition_inputs.render_video,
                    config=config,
                    ref_tensor=None,
                    use_compile=use_compile,
                    compile_mode=compile_mode,
                    force_keyframe_encode=force_keyframe_render_latent,
                )
                reference_latent = self._prepare_reference_latent(
                    condition_inputs.reference_video,
                    use_compile=use_compile,
                    compile_mode=compile_mode,
                    cast_to_vae_dtype=cast_reference_to_vae_dtype,
                )
        else:
            latents, condition = self._prepare_base_latents(
                image=condition_inputs.image,
                batch_size=batch_size,
                config=config,
                device=device,
                generator=generator,
                latents=latents,
            )
            render_latent = self._prepare_render_latent(
                condition_inputs.render_video,
                config=config,
                ref_tensor=latents,
                use_compile=use_compile,
                compile_mode=compile_mode,
                force_keyframe_encode=force_keyframe_render_latent,
            )
            reference_latent = self._prepare_reference_latent(
                condition_inputs.reference_video,
                use_compile=use_compile,
                compile_mode=compile_mode,
                cast_to_vae_dtype=cast_reference_to_vae_dtype,
            )

        return PreparedBatch(
            batch_size=batch_size,
            device=device,
            transformer_dtype=transformer_dtype,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            image_embeds=image_embeds,
            timesteps=timesteps,
            latents=latents,
            condition=condition,
            render_latent=render_latent,
            reference_latent=reference_latent,
        )

    def _build_transformer_kwargs(
        self,
        *,
        step_inputs: DenoiseStepInputs,
        condition_inputs: ConditionInputs,
        attention_kwargs,
        use_negative_prompt: bool = False,
        **extra_kwargs,
    ):
        kwargs = {
            "hidden_states": step_inputs.latent_model_input,
            "render_latent": step_inputs.render_latent,
            "render_mask": condition_inputs.render_mask,
            "camera_embedding": condition_inputs.camera_embedding,
            "timestep": step_inputs.timestep,
            "encoder_hidden_states": (
                step_inputs.negative_prompt_embeds if use_negative_prompt else step_inputs.prompt_embeds
            ),
            "encoder_hidden_states_image": step_inputs.image_embeds,
            "attention_kwargs": attention_kwargs,
            "return_dict": False,
        }
        kwargs.update(extra_kwargs)
        return kwargs

    def _apply_callback_outputs(
        self,
        *,
        callback_on_step_end,
        callback_on_step_end_tensor_inputs,
        i,
        t,
        latents,
        prompt_embeds,
        negative_prompt_embeds,
        local_values: Dict[str, Any],
    ):
        if callback_on_step_end is None:
            return latents, prompt_embeds, negative_prompt_embeds

        callback_kwargs = {k: local_values[k] for k in callback_on_step_end_tensor_inputs}
        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
        latents = callback_outputs.pop("latents", latents)
        prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
        negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)
        return latents, prompt_embeds, negative_prompt_embeds

    def _decode_or_return_latents(self, latents, *, output_type: str, return_dict: bool, use_autocast: bool = False):
        if output_type != "latent":
            latents = latents.to(self.vae.dtype)
            if use_autocast:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                    video = keyframe_vae_decode(self.vae, latents, rescale=True)
            else:
                video = keyframe_vae_decode(self.vae, latents, rescale=True)
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        if not return_dict:
            return (video,)
        return WanPipelineOutput(frames=video)

    def _run_standard_denoise_loop(
        self,
        *,
        prepared: PreparedBatch,
        condition_inputs: ConditionInputs,
        config: PipelineCallConfig,
        attention_kwargs,
        callback_on_step_end,
        callback_on_step_end_tensor_inputs,
        transformer_extra_kwargs: Optional[Dict[str, Any]] = None,
        progress_total: Optional[int] = None,
    ):
        transformer_extra_kwargs = transformer_extra_kwargs or {}
        timesteps = prepared.timesteps
        latents = prepared.latents
        prompt_embeds = prepared.prompt_embeds
        negative_prompt_embeds = prepared.negative_prompt_embeds
        num_warmup_steps = len(timesteps) - config.num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=progress_total or config.num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                latent_model_input = torch.cat([latents, prepared.condition], dim=1).to(prepared.transformer_dtype)
                timestep = t.expand(latents.shape[0])
                step_inputs = DenoiseStepInputs(
                    latent_model_input=latent_model_input,
                    timestep=timestep,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    image_embeds=prepared.image_embeds,
                    render_latent=prepared.render_latent,
                    reference_latent=prepared.reference_latent,
                )
                noise_pred = self.transformer(
                    **self._build_transformer_kwargs(
                        step_inputs=step_inputs,
                        condition_inputs=condition_inputs,
                        attention_kwargs=attention_kwargs,
                        **transformer_extra_kwargs,
                    )
                )[0]

                if self.do_classifier_free_guidance:
                    noise_uncond = self.transformer(
                        **self._build_transformer_kwargs(
                            step_inputs=step_inputs,
                            condition_inputs=condition_inputs,
                            attention_kwargs=attention_kwargs,
                            use_negative_prompt=True,
                            **transformer_extra_kwargs,
                        )
                    )[0]
                    noise_pred = noise_uncond + config.guidance_scale * (noise_pred - noise_uncond)

                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                callback_values = {
                    "latents": latents,
                    "prompt_embeds": prompt_embeds,
                    "negative_prompt_embeds": negative_prompt_embeds,
                }
                latents, prompt_embeds, negative_prompt_embeds = self._apply_callback_outputs(
                    callback_on_step_end=callback_on_step_end,
                    callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                    i=i,
                    t=t,
                    latents=latents,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    local_values=callback_values,
                )

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None
        return latents

    def _run_dmd_denoise_loop(
        self,
        *,
        prepared: PreparedBatch,
        condition_inputs: ConditionInputs,
        attention_kwargs,
        callback_on_step_end,
        callback_on_step_end_tensor_inputs,
        mode: str,
    ):
        timesteps = prepared.timesteps
        latents = prepared.latents
        prompt_embeds = prepared.prompt_embeds
        negative_prompt_embeds = prepared.negative_prompt_embeds
        condition = prepared.condition.to(prepared.transformer_dtype)
        expanded_timesteps = [t.expand(latents.shape[0]).to(prepared.device) for t in timesteps]
        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=len(timesteps)) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                latent_model_input = torch.cat([latents.to(prepared.transformer_dtype), condition], dim=1)
                timestep = expanded_timesteps[i]
                step_inputs = DenoiseStepInputs(
                    latent_model_input=latent_model_input,
                    timestep=timestep,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    image_embeds=prepared.image_embeds,
                    render_latent=prepared.render_latent,
                    reference_latent=prepared.reference_latent,
                )
                transformer_kwargs = self._build_transformer_kwargs(
                    step_inputs=step_inputs,
                    condition_inputs=condition_inputs,
                    attention_kwargs=attention_kwargs,
                    point_map_latent=None,
                    point_map_ref_latent=None,
                    reference_latent=prepared.reference_latent,
                    ref_index=condition_inputs.ref_index,
                    camera_qt=condition_inputs.camera_qt,
                    camera_qt_ref=condition_inputs.camera_qt_ref,
                )

                allow_grad = mode == "train" and i == len(timesteps) - 1
                if allow_grad:
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                        noise_pred = self.transformer(**transformer_kwargs)[0]
                else:
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                        noise_pred = self.transformer(**transformer_kwargs)[0]

                latents = self.scheduler.step(noise_pred, latents, timesteps, i)
                callback_values = {
                    "latents": latents,
                    "prompt_embeds": prompt_embeds,
                    "negative_prompt_embeds": negative_prompt_embeds,
                }
                latents, prompt_embeds, negative_prompt_embeds = self._apply_callback_outputs(
                    callback_on_step_end=callback_on_step_end,
                    callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                    i=i,
                    t=t,
                    latents=latents,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    local_values=callback_values,
                )

                progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None
        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @property
    def attention_kwargs(self):
        return self._attention_kwargs
