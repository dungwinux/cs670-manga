from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import os 
import pandas as pd 
import torch
from tqdm import tqdm

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-72B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-72B-Instruct")


#--------------
#   Running
#--------------
df = pd.read_csv("data/annotation/annotation_cleaned.csv")
format_text = open("prompts/format_json.md", 'r').read()

output_file = 'data/annotation/qwen_72.csv'
# Initialize CSV file with headers if it doesn’t exist yet
if not os.path.exists(output_file):
    pd.DataFrame(columns=['image_path', 'outputs']).to_csv(output_file, index=False)

paths = sorted(list(set(df.image_path.tolist())))
for i, image_path in enumerate(tqdm(paths)): 
    df_specific = df[df['image_path']==image_path].reset_index(drop=True)
    if len(df_specific) == 1 and len(df_specific.text_ja.tolist()[0]) <=1: 
        continue
    if image_path in pd.read_csv(output_file).image_path.tolist():
        print("Skipping", image_path)
        continue
    page_spec = ""
    for j in range(len(df_specific)): 
        page_spec += format_text.format(i=str(j), orig=df_specific.text_ja.tolist()[j]) + "\n\n"
    messages = [{
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image_path,
            },
            {
                "type": "text", 
                "text": open("prompts/translation_json.md").read().format(orig=page_spec)},
        ],
    }]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    
    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=5000, do_sample=False)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    # Save incrementally
    pd.DataFrame({'image_path': [image_path], 'outputs': [output_text[0]]}).to_csv(output_file, mode='a', index=False, header=False)

