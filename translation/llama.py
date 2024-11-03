import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

model_id = "meta-llama/Llama-3.2-90B-Vision-Instruct"

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

#--------------
#   Running
#--------------
df = pd.read_csv("data/annotation/annotation_cleaned.csv")
format_text = open("prompts/format_json.md", 'r').read()

output_file = 'data/annotation/llama_90.csv'
# Initialize CSV file with headers if it doesn’t exist yet
if not os.path.exists(output_file):
    pd.DataFrame(columns=['image_path', 'outputs']).to_csv(output_file, index=False)

paths = sorted(list(set(df.image_path.tolist())))
for i, image_path in enumerate(tqdm(paths)): 
    df_specific = df[df['image_path']==image_path].reset_index(drop=True)
    if len(df_specific) == 1 and len(df_specific.text_ja.tolist()[0]) <=1: 
        continue
    page_spec = ""
    for j in range(len(df_specific)): 
        page_spec += format_text.format(i=str(j), orig=df_specific.text_ja.tolist()[j]) + "\n\n"
    messages = [{
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": Image.open(requests.get(image_path, stream=True).raw),
            },
            {
                "type": "text", 
                "text": open("prompts/translation_json.md").read().format(orig=page_spec)},
        ],
    }]
    
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=2000, do_sample=False)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    # Save incrementally
    pd.DataFrame({'image_path': [image_path], 'outputs': [output_text[0]]}).to_csv(output_file, mode='a', index=False, header=False)
    print(output_text[0])









input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(
    image,
    input_text,
    add_special_tokens=False,
    return_tensors="pt",
).to(model.device)

output = model.generate(**inputs, max_new_tokens=30)
print(processor.decode(output[0]))
