
import torch
import os
import itertools
import random
import multiprocessing
import re
from diffusers import DiffusionPipeline

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)  
    
    
D
gpu_ids = list(range(torch.cuda.device_count()))
print(f"Using GPUs: {gpu_ids}")


model_id = "stabilityai/stable-diffusion-3-medium-diffusers"


save_path = "/"
os.makedirs(save_path, exist_ok=True)


prompt_file = os.path.join(save_path, "prompts.txt")


num_images = 
batch_size = 1  


ages = ["5", "10", "15", "20", "25", "30", "40", "50", "60", "70", "80"]
ethnicities = ["Caucasian", "African", "Asian", "Hispanic", "Middle Eastern", "South Asian", "Native American", "Pacific Islander"]
genders = ["man", "woman"]


combinations = list(itertools.product(ages, ethnicities, genders))


random.seed(42)


existing_images = sorted(
    [int(re.search(r"image_(\d+).png", f).group(1)) for f in os.listdir(save_path) if re.match(r"image_\d+\.png", f)]
)


all_indices = set(range(num_images))
existing_indices = set(existing_images)
missing_indices = sorted(list(all_indices - existing_indices))  

print(f"Resuming missing images: {len(missing_indices)} remaining")


negative_prompt = "full body, hands, arms, legs, blurry, low quality, disfigured, deformed, extra limbs, watermark, text, background, artistic, painting, cartoon, 3d, sketch"


random.shuffle(combinations)


lock = multiprocessing.Lock()



def generate_images_on_gpu(gpu_id, image_indices):



    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.to(f"cuda:{gpu_id}")

    local_prompts = []  

    for image_count in image_indices:
        batch_prompts = []
        current_batch_size = min(batch_size, len(image_indices) - image_indices.index(image_count))

        for j in range(current_batch_size):
            idx = image_indices.index(image_count) + j
            if idx >= len(image_indices): 
                break
            actual_index = image_indices[idx]

            age, ethnicity, gender = combinations[actual_index % len(combinations)]
            prompt = f"close-up portrait of a {age} years old {ethnicity} {gender}, highly detailed, ultra realistic, hyper detailed face, studio lighting, sharp focus, professional photography, 8k"
            batch_prompts.append(prompt)


        images = pipe(prompt=batch_prompts, negative_prompt=negative_prompt).images


        for idx, img in enumerate(images):
            actual_index = image_indices[image_indices.index(image_count) + idx]
            img_name = f"image_{actual_index:05d}.png"
            img.save(os.path.join(save_path, img_name))

         
            local_prompts.append(f"{img_name}: {batch_prompts[idx]}\n")

        print(f"GPU {gpu_id}: {image_count}번째 이미지 완료")

    with lock:
        with open(prompt_file, "a") as f:
            f.writelines(local_prompts)



if __name__ == "__main__":
    processes = []
    num_gpus = len(gpu_ids)

    images_per_gpu = len(missing_indices) // num_gpus
    leftover_images = len(missing_indices) % num_gpus

    for i, gpu_id in enumerate(gpu_ids):
        start_idx = i * images_per_gpu
        end_idx = start_idx + images_per_gpu
        if i == num_gpus - 1:
            end_idx += leftover_images  

        missing_subset = missing_indices[start_idx:end_idx]  
        p = multiprocessing.Process(target=generate_images_on_gpu, args=(gpu_id, missing_subset))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

