import torch
import os
import itertools
import random
import multiprocessing
from PIL import Image
from diffusers import StableDiffusionXLImg2ImgPipeline

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)


gpu_ids = list(range(torch.cuda.device_count()))
print(f"Using GPUs: {gpu_ids}")


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

model_id = "stabilityai/stable-diffusion-xl-base-1.0"

input_image_path = "/"
save_path = "/"
os.makedirs(save_path, exist_ok=True)


prompt_file = os.path.join(save_path, "prompts.txt")


num_images = 25000
batch_size = 1


ages = ["5", "10", "15", "20", "25", "30", "40", "50", "60", "70", "80"]
ethnicities = ["Caucasian", "African", "Asian", "Hispanic", "Middle Eastern", "South Asian", "Native American", "Pacific Islander"]
genders = ["man", "woman"]
combinations = list(itertools.product(ages, ethnicities, genders))


random.seed(42)
random.shuffle(combinations)


all_input_images = sorted([
    f for f in os.listdir(input_image_path)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])
input_image_map = {
    idx: fname for idx, fname in enumerate(all_input_images[:num_images])
}


all_indices = set(input_image_map.keys())


existing_images = sorted([
    int(f.split("_")[1].split(".")[0])
    for f in os.listdir(save_path) if f.startswith("image_") and f.endswith(".png")
])
missing_indices = sorted(list(all_indices - set(existing_images)))



negative_prompt = (
    "full body, hands, arms, legs, blurry, low quality, disfigured, deformed, "
    "extra limbs, watermark, text, background, artistic, painting, cartoon, 3d, sketch"
)

lock = multiprocessing.Lock()


def generate_i2i_on_gpu(gpu_id, image_indices):
   

    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        variant="fp16"
    ).to(f"cuda:{gpu_id}")
    pipe.enable_attention_slicing()

    local_prompts = []

    for image_count in image_indices:
        try:
            init_filename = input_image_map[image_count]
            init_image_path = os.path.join(input_image_path, init_filename)
            init_image = Image.open(init_image_path).convert("RGB").resize((1024, 1024))
        except Exception as e:
        
            continue


        age, ethnicity, gender = combinations[image_count % len(combinations)]
        prompt = (
            f"close-up portrait of a {age} years old {ethnicity} {gender}, "
            f"highly detailed, ultra realistic, hyper detailed face, "
            f"studio lighting, sharp focus, professional photography, 8k"
        )


        try:
            images = pipe(
                prompt=[prompt],
                negative_prompt=negative_prompt,
                image=init_image,
                strength=0.7,
                guidance_scale=7,
                num_inference_steps=28
            ).images
        except Exception as e:

            continue


        output_filename = f"image_{image_count:05d}.png"
        images[0].save(os.path.join(save_path, output_filename))
        local_prompts.append(
            f"input: {init_filename} | prompt: {prompt} | output: {output_filename}\n"
        )

        images[0].close()
        init_image.close()
        torch.cuda.empty_cache()
        print(f"GPU {gpu_id}: {output_filename} complete")

    with lock:
        with open(prompt_file, "a") as f:
            f.writelines(local_prompts)

    del pipe
    torch.cuda.empty_cache()


if __name__ == "__main__":
    processes = []
    num_gpus = len(gpu_ids)

    images_per_gpu = len(missing_indices) // num_gpus
    leftover = len(missing_indices) % num_gpus

    for i, gpu_id in enumerate(gpu_ids):
        start = i * images_per_gpu
        end = start + images_per_gpu + (leftover if i == num_gpus - 1 else 0)
        assigned_indices = missing_indices[start:end]
        p = multiprocessing.Process(target=generate_i2i_on_gpu, args=(gpu_id, assigned_indices))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print("ALL i2i imaage generation complete!")
