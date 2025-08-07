

import torch
import os
import itertools
import random
import multiprocessing
import re
from PIL import Image
from diffusers import StableDiffusion3Img2ImgPipeline


gpu_ids = list(range(torch.cuda.device_count()))
print(f" Using GPUs: {gpu_ids}")


model_id = "stabilityai/stable-diffusion-3-medium-diffusers"


input_folder = "/"
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


input_image_map = {}
all_filenames = [f for f in os.listdir(input_folder) if re.search(r"(\d+)", f)]
selected_files = random.sample(all_filenames, min(num_images, len(all_filenames)))

for i, filename in enumerate(selected_files):
    input_image_map[i] = filename  


existing_images = sorted(
    [int(re.search(r"image_(\d+).png", f).group(1)) for f in os.listdir(save_path) if re.match(r"image_\d+\.png", f)]
)
all_indices = set(range(num_images))
missing_indices = sorted(list(all_indices - set(existing_images)))
print(f"🔄 Resuming missing images: {len(missing_indices)} remaining")

# Negative prompt
negative_prompt = "full body, hands, arms, legs, blurry, low quality, disfigured, deformed, extra limbs, watermark, text, background, artistic, painting, cartoon, 3d, sketch"


lock = multiprocessing.Lock()

def get_input_image(actual_index):
    if actual_index in input_image_map:
        input_image_path = os.path.join(input_folder, input_image_map[actual_index])
        try:
            return Image.open(input_image_path).convert("RGB"), input_image_path
        except Exception as e:
            print(f"Error opening image {input_image_path}: {e}")
            return None, None
    else:
        print(f"Missing file for index {actual_index}")
        return None, None


def generate_images_on_gpu(gpu_id, image_indices):


    if gpu_id >= torch.cuda.device_count():
        print(f"Invalid GPU ID: {gpu_id}. Available GPUs: {torch.cuda.device_count()}")
        return

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

 
    pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.to(f"cuda:{gpu_id}")

    for image_count in image_indices:
        age, ethnicity, gender = combinations[image_count % len(combinations)]
        prompt = f"close-up portrait of a {age} years old {ethnicity} {gender}, highly detailed, ultra realistic, hyper detailed face, studio lighting, sharp focus, professional photography, 8k"

        input_image, input_image_path = get_input_image(image_count)
        if input_image is None:
            continue

        torch.cuda.empty_cache()  
        try:
            result = pipe(
                prompt=prompt,
                image=input_image,
                strength=0.7,
                guidance_scale=7.0,
                num_inference_steps=28,
                negative_prompt=negative_prompt
            )
        except Exception as e:
            print(f"Generation failed for index {image_count}: {e}")
            continue

        output_image = result.images[0]
        img_name = f"image_{image_count:05d}.png"
        output_path = os.path.join(save_path, img_name)
        output_image.save(output_path)

        with lock:
            with open(prompt_file, "a") as f:
                f.write(f"{img_name}: {prompt} (Input: {os.path.basename(input_image_path)})\n")



if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    processes = []
    num_gpus = len(gpu_ids)

    images_per_gpu = len(missing_indices) // num_gpus
    leftover = len(missing_indices) % num_gpus

    for i, gpu_id in enumerate(gpu_ids):
        start_idx = i * images_per_gpu
        end_idx = start_idx + images_per_gpu
        if i == num_gpus - 1:
            end_idx += leftover

        subset = missing_indices[start_idx:end_idx]
        p = multiprocessing.Process(target=generate_images_on_gpu, args=(gpu_id, subset))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

