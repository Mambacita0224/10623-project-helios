# Example: Running inference with 2-GPU parallelism
# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 infer_helios.py \
#     --enable_parallelism \
#     --cp_backend "ulysses" \   #  ["ring", "ulysses", "unified", "ulysses_anything"]

CUDA_VISIBLE_DEVICES=0 python infer_helios.py \
    --base_model_path "BestWishYsh/Helios-Mid" \
    --transformer_path "BestWishYsh/Helios-Mid" \
    --sample_type "v2v" \
    --num_frames 99 \
    --fps 24 \
    --video_path "example/car.mp4" \
    --video_noise_sigma_min 0.111 \
    --video_noise_sigma_max 0.135 \
    --prompt "A bright yellow Lamborghini Huracn Tecnica speeds along a curving mountain road, surrounded by lush green trees under a partly cloudy sky. The car's sleek design and vibrant color stand out against the natural backdrop, emphasizing its dynamic movement. The road curves gently, with a guardrail visible on one side, adding depth to the scene. The motion blur captures the sense of speed and energy, creating a thrilling and exhilarating atmosphere. A front-facing shot from a slightly elevated angle, highlighting the car's aggressive stance and the surrounding greenery." \
    --guidance_scale 5.0 \
    --is_enable_stage2 \
    --pyramid_num_inference_steps_list 20 20 20 \
    --use_zero_init \
    --zero_steps 1 \
    --enable_compile \
    --output_folder "./output_helios/helios-mid"


    # --enable_low_vram_mode \
    # --group_offloading_type "leaf_level" \  # ["leaf_level", "block_level"]
    # --num_blocks_per_group 4 \
    # --pyramid_num_inference_steps_list 17 17 17 \