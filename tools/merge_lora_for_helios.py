import sys
from argparse import Namespace


sys.path.append("../")
from helios.modules.transformer_helios import HeliosTransformer3DModel
from helios.pipelines.pipeline_helios import HeliosPipeline
from helios.utils.utils_base import load_extra_components


transformer_additional_kwargs = {
    "has_multi_term_memory_patch": True,
    "zero_history_timestep": True,
    "guidance_cross_attn": True,
    "restrict_self_attn": False,
    "is_train_restrict_lora": False,
    "restrict_lora": False,
    "restrict_lora_rank": 128,
}

transformer = HeliosTransformer3DModel.from_pretrained(
    "1_formal_ckpts/ablation_stage3_2_mid-train_v4_e2500-ema",
    subfolder="transformer",
    transformer_additional_kwargs=transformer_additional_kwargs,
)
pipe = HeliosPipeline.from_pretrained(
    "Wan-AI/Wan2.1-T2V-14B-Diffusers",
    transformer=transformer,
)

pipe.load_lora_weights(
    "ablation_stage3_3_post-train-emergency_only-gan/checkpoint-2000/model_ema/pytorch_lora_weights.safetensors",
    adapter_name="default",
)
pipe.set_adapters(["default"], adapter_weights=[1.0])


args = Namespace()
if not hasattr(args, "training_config"):
    args.training_config = Namespace()
args.training_config.is_enable_stage1 = True
args.training_config.restrict_self_attn = True
args.training_config.is_amplify_history = True
args.training_config.is_use_gan = True
load_extra_components(
    args,
    transformer,
    "ablation_stage3_3_post-train-emergency_only-gan/checkpoint-2000/model_ema/transformer_partial.pth",
)

pipe.fuse_lora()
pipe.unload_lora_weights()
pipe.transformer.save_pretrained(
    "1_formal_ckpts/ablation_stage3_3_post-train-emergency_only-gan_e2000-ema/transformer"
)
