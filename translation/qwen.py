from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import os
import pandas as pd
import torch
from tqdm import tqdm
import re
import argparse

def extract_tag_text(text, tag, random=False):
    '''
    Extract text between tags
    '''
    if random:
        pattern = re.compile(rf'<{tag}>(.*?)<(.*?)>', re.DOTALL)
    else:
        pattern = re.compile(rf'<{tag}>(.*?)</{tag}>', re.DOTALL)
    
    matches = pattern.findall(text)
    
    return matches[0] if matches else None


def single_inference(model, processor, page_spec, path, do_sample=False):
    messages = [{
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": path,
            },
            {
                "type": "text", 
                "text": open("prompts/translation_json.md").read().format(orig=page_spec),
            },
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")
    
    generated_ids = model.generate(**inputs, max_new_tokens=3000, do_sample=do_sample)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]


def check_inference(output_text, df_specific):
    '''
    Verify if the output has the correct number of text-translation pairs and matches original Japanese text
    '''
    pairs = re.findall(r'<text>(.*?)</text>\s*<translation>(.*?)</translation>', output_text, re.DOTALL)
    
    # Check if the count of pairs matches the ground truth
    if len(pairs) != len(df_specific):
        print("Mismatched number of pairs")
        print(output_text)
        return False
    
    # Iterate through pairs and verify each against the ground truth
    for i, (text, translation) in enumerate(pairs):
        original_text = text.strip()
        ground_truth_text = df_specific.text.iloc[i].strip()
        
        # Check if the original Japanese text matches
        if original_text != ground_truth_text:
            print("Original text not matching")
            print(f"Output: {original_text}")
            print(f"Ground Truth: {ground_truth_text}")
            return False

    return True


def inference(model, processor, df, output_file):
    '''
    Run inference over the dataset
    '''
    paths = sorted(list(set(df.path.tolist())))
    
    for path in tqdm(paths): 
        # Get a subset of the data that correspond to the book path in question 
        df_specific = df[df['path'] == path].reset_index(drop=True)
        
        if len(df_specific) == 1 and len(df_specific.text.iloc[0]) <= 1:
            print("Empty text, skipping", path)
            continue
        if os.path.exists(output_file) and path in pd.read_csv(output_file).path.tolist():
            print("Skipping", path)
            continue

        page_spec = "\n\n".join(
            [f"<item_{j}>    <text>    {df_specific.text.iloc[j]}    </text></item_{j}>"
             for j in range(len(df_specific))]
        )

        output_text = single_inference(model, processor, page_spec, path).replace('\n', '')

        retry_count = 0
        while not check_inference(output_text, df_specific):
            print("Retrying", path)
            output_text = single_inference(model, processor, page_spec + "\n\n Please follow the output template strictly.", path, do_sample=True).replace('\n', '')
            retry_count += 1
            if retry_count > 5:
                print(output_text)
                print("Failed to get correct output after 5 retries. Skipping", path)
                break
        
        if check_inference(output_text, df_specific): 
            df_specific['text_translated'] = [
                (extracted_translation.strip() if extracted_translation else "") 
                for extracted_translation in [
                    extract_tag_text(
                        extract_tag_text(output_text, f'item_{j}') or "", 'translation'
                    )
                    for j in range(len(df_specific))
                ]
            ]
        else: 
            df_specific['text_translated'] = [
                "" for j in range(len(df_specific))
            ]
        df_specific['outputs'] = [output_text] * len(df_specific)
        df_specific['text_original'] = df_specific.text
        df_specific['path'] = path

        df_specific.to_csv(output_file, mode='a', index=False, header=False)
        print("Translated", path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="Model name")
    args = parser.parse_args()
    model_name = args.model_name
    print("Inferencing with:", model_name)

    if model_name == "qwen_72":
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-72B-Instruct",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
            cache_dir="/scratch3/workspace/ctpham_umass_edu-ft/.cache"
        )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-72B-Instruct")
        output_file = '../data/output/qwen_72b_openmantra.csv'
    elif model_name == "qwen_7":
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-7B-Instruct",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        output_file = '../data/output/qwen_7b_openmantra.csv'

    # Read in dataframe ----
    df = pd.read_csv("../data/output/detection/merged.csv")
    if not os.path.exists(output_file):
        pd.DataFrame(columns=['path', 'coordinates', 'outputs', 'text_translated']).to_csv(output_file, index=False)

    inference(model, processor, df, output_file)
