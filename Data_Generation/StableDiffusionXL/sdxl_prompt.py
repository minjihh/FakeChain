import torch
import os
import itertools
import random
import multiprocessing
import re
from diffusers import StableDiffusionXLPipeline
from PIL import Image

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)


gpu_ids = list(range(torch.cuda.device_count()))
print(f"Using GPUs: {gpu_ids}")

model_id = "stabilityai/stable-diffusion-xl-base-1.0"


save_path = "/"
os.makedirs(save_path, exist_ok=True)


prompt_file = os.path.join(save_path, "prompts.txt")


start_index = 15000
end_index = 24999 


batch_size = 1


ages = ["5", "10", "15", "20", "25", "30", "40", "50", "60", "70", "80"]
ethnicities = [
    "Caucasian", "African", "Asian", "Hispanic",
    "Middle Eastern", "South Asian", "Native American", "Pacific Islander"
]
genders = ["man", "woman"]
combinations = list(itertools.product(ages, ethnicities, genders))


random.seed(42)
random.shuffle(combinations)


existing_images = sorted([
    int(re.search(r"image_(\d+).png", f).group(1))
    for f in os.listdir(save_path) if re.match(r"image_\d+\.png", f)
])
all_indices = set(range(start_index, end_index + 1))
missing_indices = sorted(list(all_indices - set(existing_images)))

print(f"Resuming missing images: {len(missing_indices)} remaining")


negative_prompt = (
    "full body, hands, arms, legs, blurry, low quality, disfigured, deformed, "
    "extra limbs, watermark, text, background, artistic, painting, cartoon, 3d, sketch"
)


lock = multiprocessing.Lock()


def generate_images_on_gpu(gpu_id, image_indices):
  

    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        variant="fp16"
    ).to(f"cuda:{gpu_id}")
    pipe.enable_attention_slicing()
    pipe.enable_vae_tiling()
    pipe.enable_model_cpu_offload()

    for image_count in image_indices:
        batch_prompts = []
        current_batch_size = min(batch_size, len(image_indices) - image_indices.index(image_count))

        for j in range(current_batch_size):
            idx = image_indices.index(image_count) + j
            if idx >= len(image_indices):
                break
            actual_index = image_indices[idx]

            age, ethnicity, gender = combinations[actual_index % len(combinations)]
            prompt = (
                f"close-up portrait of a {age} years old {ethnicity} {gender}, "
                f"highly detailed, ultra realistic, hyper detailed face, "
                f"studio lighting, sharp focus, professional photography, 8k"
            )
            batch_prompts.append(prompt)


        images = pipe(
            prompt=batch_prompts,
            negative_prompt=negative_prompt,
            num_inference_steps=28,
            guidance_scale=7
        ).images

        for idx, img in enumerate(images):
            actual_index = image_indices[image_indices.index(image_count) + idx]
            img_name = f"image_{actual_index:05d}.png"
            img.save(os.path.join(save_path, img_name))

        
            prompt_log = f"{img_name}: {batch_prompts[idx]}\n"
            with lock:
                with open(prompt_file, "a") as f:
                    f.write(prompt_log)



if __name__ == "__main__":
    processes = []
    num_gpus = len(gpu_ids)

    images_per_gpu = len(missing_indices) // num_gpus
    leftover = len(missing_indices) % num_gpus

    for i, gpu_id in enumerate(gpu_ids):
        start = i * images_per_gpu
        end = start + images_per_gpu + (leftover if i == num_gpus - 1 else 0)
        assigned_indices = missing_indices[start:end]
        p = multiprocessing.Process(target=generate_images_on_gpu, args=(gpu_id, assigned_indices))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

