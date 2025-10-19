from transformers import AutoModel, GenerationConfig
from typing import Union
from prompts_final import prompts, configs
from tqdm import tqdm
import pandas as pd
import torch
import numpy as np
import os
import re
from torchvision import transforms
import time
from PIL import Image
from utils import extract_cropped_human_frames, check_tags, parse_to_dict, clean_output, load_frames_from_directory, pad_frames
import glob
import ast

def triage_trauma(video: Union[list, str], model, f):

    frames = []
    if isinstance(video, list):
        frames = video
    elif os.path.isdir(video):
        frames = load_frames_from_directory(video)
    elif os.path.isfile(video):
        frames = extract_cropped_human_frames(video)
    else:
        raise ValueError("Invalid video input")
        
    if len(frames) == 0:
        raise ValueError("No frames found")
        
    # Pad frames to multiple of 8
    frames = pad_frames(frames)
    
    # Configure model
    model.config.num_video_frames, model.config.fps = len(frames), 0
    model.config.llm_cfg['use_bfloat16'] = True
    
    # Generate report
    if isinstance(f ,str):
        return forward_pass(frames, model, f)
    else:
        return forward_pass(frames, model)
        

def triage_casualty(frames: list, model, config=None, input_prompt=None, generation_config=GenerationConfig(max_new_tokens=256), sys_prompt=None):
    model.config.llm_cfg['max_length'] = config["max_length"]
    model.config.llm_cfg['min_length'] = config["min_length"]
    model.config.llm_cfg['temperature'] = config["temperature"]
    system_prompt_thinking = sys_prompt
    generation_config.max_new_tokens = config["max_tokens"]
    generation_config.repetition_penalty = config.get("repetition_penalty", 1.0)
    use_thinking = config.get("use_thinking", False)
    if use_thinking:
        prompt = system_prompt_thinking.format(question=input_prompt)
    else:
        prompt = input_prompt
    print("Prompt:", prompt)
    response = model.generate_content([prompt ,frames], generation_config=generation_config)
    return response

def forward_pass(frames: list, model, video=""):
    start_time = time.time()
    classes_trauma = ["normal", "wound", "amputation", "not testable"]
    classes_sh = ["present", "absent"]
    generation_config = model.default_generation_config
    generation_config = GenerationConfig(
        max_new_tokens = 1024,  # Will be updated based on context
        max_length = generation_config.max_length,
        pad_token_id = generation_config.pad_token_id,
        bos_token_id = generation_config.bos_token_id,
        eos_token_id = generation_config.eos_token_id,
        # do_sample=True,
        # temperature = 0.5,
    )

    ## prompts

    prompts_trauma = prompts()
    configs_trauma = configs()

    system_prompt_thinking = prompts_trauma.thinking()
    description_prompt = prompts_trauma.description()
    trauma_report_prompt = prompts_trauma.trauma_report()
    trauma_regen_prompt = prompts_trauma.trauma_regen()
    verify_amputation_prompt = prompts_trauma.verify_amputation()
    sh_ma_description_prompt = prompts_trauma.description_sh()
    sh_ma_report_prompt = prompts_trauma.report_sh()
    sh_ma_regen_prompt = prompts_trauma.regen_sh()
    trauma_pattern = prompts_trauma.regex()
    sh_ma_pattern = prompts_trauma.regex_sh()

    ## configs
    trauma_description_config = configs_trauma.trauma_desc()
    trauma_desc_regen_config = configs_trauma.trauma_desc_regen()
    standard_report_config = configs_trauma.standard_report()
    sh_ma_description_config = configs_trauma.sh_desc()
    sh_ma_description_regen_config = configs_trauma.sh_desc_regen()

    print("\n\n######### INITIAL DESCRIPTION PASS #########\n\n")
    description = triage_casualty(frames, model, trauma_description_config, description_prompt, generation_config, system_prompt_thinking)
    print("Description:", description)

    ## Check for incomplete description
    if check_tags(description)==False:
        print("Description format incorrect, regenerating...")
        description = triage_casualty(frames, model, trauma_desc_regen_config, description_prompt, generation_config, system_prompt_thinking)
        print("Regenerated Description:", description)
    else:
        print("Description format correct.")

    print("\n\n######### INITIAL TRAUMA REPORT PASS #########\n\n")
    report_description = str(description).replace("<think>", "").replace("</think>", "").replace("<answer>", "").replace("</answer>", "")
    trauma_report = triage_casualty(frames, model, standard_report_config, "Video Description: " + report_description + " " + trauma_report_prompt, generation_config, system_prompt_thinking)
    print("Trauma Report:", trauma_report)

    ## Report JSON Format Check
    trauma_report = trauma_report.lower().replace("\n", "").strip().replace('"', "").replace("'", "")
    print("formatted report", trauma_report)
    if bool(trauma_pattern.match(trauma_report))==False:
        print("Report format incorrect, regenerating...")
        trauma_report = triage_casualty(frames, model, standard_report_config, "Previous Response " + trauma_report + " " + trauma_regen_prompt, generation_config, system_prompt_thinking)
        trauma_report = trauma_report.lower().replace("\n", "").strip().replace('"', "").replace("'", "")
        print("Regenerated Trauma Report:", trauma_report)
        if bool(trauma_pattern.match(trauma_report)):
            # print("Valid after regen -> ", end="")
            pass
        else:
            print("\nFailed trauma report validation")
            print("Did not pass the check", trauma_report)
            trauma_report = "{head: _, torso: _, upper extremity: _, lower extremity: _}"
            print("Defaulting to -> ", trauma_report)
    
    ## Amputation Verification
    trauma_report2 = trauma_report
    trauma_report = clean_output(parse_to_dict(trauma_report), classes_trauma) #converting standard dictionary and cleaning
    
    if (trauma_report["upper extremity"] == "wound") or (trauma_report["lower extremity"] == "wound"):
        print("\n\n######### AMPUTATION VERIFICATION PASS #########\n\n")
        trauma_report = triage_casualty(frames, model, standard_report_config, "Current Report: " + str(trauma_report) + " " + verify_amputation_prompt, generation_config, system_prompt_thinking)
        trauma_report = trauma_report.lower().replace("\n", "").strip().replace('"', "").replace("'", "")
    
    if bool(trauma_pattern.match(trauma_report))==False:
        print("Report format incorrect, regenerating...")
        trauma_report = triage_casualty(frames, model, standard_report_config, "Previous Response " + trauma_report + " " + trauma_regen_prompt, generation_config, system_prompt_thinking)
        trauma_report = trauma_report.lower().replace("\n", "").strip().replace('"', "").replace("'", "")
        print("Regenerated Trauma Report:", trauma_report)
        if bool(trauma_pattern.match(trauma_report)):
            # print("Valid after regen -> ", end="")
            pass
        else:
            print("\nFailed trauma report validation")
            print("Did not pass the check", trauma_report)
            trauma_report = trauma_report2
            print("Defaulting to -> ", trauma_report)


    ## SH and MA
    print("\n\n######### SEVERE HEMORRHAGE AND MOTOR ALERTNESS PASS #########\n\n")
    sh_ma_description = triage_casualty(frames, model, sh_ma_description_config, sh_ma_description_prompt, generation_config, system_prompt_thinking)
    print("SH and MA Description:", sh_ma_description)
    if check_tags(sh_ma_description)==False:
        print("Description format incorrect, regenerating...")
        sh_ma_description = triage_casualty(frames, model, sh_ma_description_regen_config, sh_ma_description_prompt, generation_config, system_prompt_thinking)
        print("Regenerated SH and MA Description:", sh_ma_description)
    else:
        print("Description format correct.") 
    
    print("\n\n######### SEVERE HEMORRHAGE AND MOTOR ALERTNESS REPORT PASS #########\n\n")
    report_sh_ma = triage_casualty(frames, model, standard_report_config, "Video Description: " + str(sh_ma_description).replace("<think>", "").replace("</think>", "").replace("<answer>", "").replace("</answer>", "") + " " + sh_ma_report_prompt, generation_config, system_prompt_thinking)
    print("SH and MA Report:", report_sh_ma)
    report_sh_ma = report_sh_ma.lower().replace("\n", "").strip().replace('"', "").replace("'", "")
    if bool(sh_ma_pattern.match(report_sh_ma))==False:
        print("Report format incorrect, regenerating...")
        report_sh_ma = triage_casualty(frames, model, standard_report_config, "Previous Response " + report_sh_ma + " " + sh_ma_regen_prompt, generation_config, system_prompt_thinking)
        report_sh_ma = report_sh_ma.lower().replace("\n", "").strip().replace('"', "").replace("'", "")
        print("Regenerated SH and MA Report:", report_sh_ma)
        if bool(sh_ma_pattern.match(report_sh_ma)):
            # print("Valid after regen -> ", end="")
            pass
        else:
            print("\nFailed SH and MA report validation")
            print("Did not pass the check")
            report_sh_ma = "{motor alertness: _, severe hemorrhage: _}"
            print("Defaulting to -> ", report_sh_ma)
    sh_ma_report = clean_output(parse_to_dict(report_sh_ma), classes_sh) #converting standard dictionary and cleaning
    end_time = time.time()
    print(f"\n\nTotal inference time: {end_time - start_time:.2f} seconds")
    final_report = {**trauma_report, **sh_ma_report}
    print("\n\n######### FINAL REPORT #########\n\n")
    print(final_report)
    print(f"\nTotal forward pass time: {end_time - start_time:.2f} seconds")
    new_df=pd.DataFrame([final_report])
    new_df.to_csv('./final_runs.csv',mode='a',index=False,header=False)
    return final_report

if __name__ == "__main__":
    model_path = "Efficient-Large-Model/LongVILA-R1-7B"
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, device_map="auto")
    video_dir = "./casualty_videos/"
    save_dir = "arsh_temp_run"
    os.makedirs(save_dir, exist_ok=True)
    output_csv_path = './arsh_temp_run4.csv'
    processed_videos = []
    if os.path.exists(output_csv_path):
        try:
            df = pd.read_csv(output_csv_path, header=None)
            processed_videos = df[5].unique().tolist()
            print(f"Found {len(processed_videos)} videos already processed. Skipping them.")
        except Exception as e:
            print(f"Error reading existing CSV: {e}. Starting from scratch.")
    
    
    all_videos = os.listdir(video_dir)
    print(f"Total videos to check: {len(all_videos)}")

    for idx, f in enumerate(all_videos):
        if f in processed_videos:
            print(f"Skipping video: {f} (already processed)")
            continue

        print(f"Processing video: {f}")

        frames = extract_cropped_human_frames(os.path.join(video_dir, f), 16)
        
        video_name = os.path.splitext(f)[0]
        video_dir_path = os.path.join(save_dir, video_name)
        os.makedirs(video_dir_path, exist_ok=True)
        
        # Save each frame
        for frame_idx, frame in enumerate(frames):
            save_path = os.path.join(video_dir_path, f"{frame_idx}.jpg")
            try:
                frame.save(save_path, "JPEG")
                print(f"Saved frame {frame_idx} to {save_path}")
            except Exception as e:
                print(f"Error saving frame {frame_idx} from video {f}: {str(e)}")
        
        print(triage_trauma(frames, model, f))
        print("\n\n\n")



