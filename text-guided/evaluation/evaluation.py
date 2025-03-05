import json
import argparse
import os
import numpy as np
from PIL import Image
import csv
from matrics_calculator import MetricsCalculator

def mask_decode(encoded_mask,image_shape=[512,512]):
    length=image_shape[0]*image_shape[1]
    mask_array=np.zeros((length,))
    
    for i in range(0,len(encoded_mask),2):
        splice_len=min(encoded_mask[i+1],length-encoded_mask[i])
        for j in range(splice_len):
            mask_array[encoded_mask[i]+j]=1
            
    mask_array=mask_array.reshape(image_shape[0], image_shape[1])
    # to avoid annotation errors in boundary
    mask_array[0,:]=1
    mask_array[-1,:]=1
    mask_array[:,0]=1
    mask_array[:,-1]=1
            
    return mask_array

def calculate_metric(metrics_calculator,metric, src_image, tgt_image, src_mask, tgt_mask,src_prompt,tgt_prompt):
    if metric=="psnr":
        return metrics_calculator.calculate_psnr(src_image, tgt_image, None, None)
    if metric=="lpips":
        return metrics_calculator.calculate_lpips(src_image, tgt_image, None, None)
    if metric=="mse":
        return metrics_calculator.calculate_mse(src_image, tgt_image, None, None)
    if metric=="ssim":
        return metrics_calculator.calculate_ssim(src_image, tgt_image, None, None)
    if metric=="structure_distance":
        return metrics_calculator.calculate_structure_distance(src_image, tgt_image, None, None)
    if metric=="psnr_unedit_part":
        if (1-src_mask).sum()==0 or (1-tgt_mask).sum()==0:
            return "nan"
        else:
            return metrics_calculator.calculate_psnr(src_image, tgt_image, 1-src_mask, 1-tgt_mask)
    if metric=="lpips_unedit_part":
        if (1-src_mask).sum()==0 or (1-tgt_mask).sum()==0:
            return "nan"
        else:
            return metrics_calculator.calculate_lpips(src_image, tgt_image, 1-src_mask, 1-tgt_mask)
    if metric=="mse_unedit_part":
        if (1-src_mask).sum()==0 or (1-tgt_mask).sum()==0:
            return "nan"
        else:
            return metrics_calculator.calculate_mse(src_image, tgt_image, 1-src_mask, 1-tgt_mask)
    if metric=="ssim_unedit_part":
        if (1-src_mask).sum()==0 or (1-tgt_mask).sum()==0:
            return "nan"
        else:
            return metrics_calculator.calculate_ssim(src_image, tgt_image, 1-src_mask, 1-tgt_mask)
    if metric=="structure_distance_unedit_part":
        if (1-src_mask).sum()==0 or (1-tgt_mask).sum()==0:
            return "nan"
        else:
            return metrics_calculator.calculate_structure_distance(src_image, tgt_image, 1-src_mask, 1-tgt_mask)
    if metric=="psnr_edit_part":
        if src_mask.sum()==0 or tgt_mask.sum()==0:
            return "nan"
        else:
            return metrics_calculator.calculate_psnr(src_image, tgt_image, src_mask, tgt_mask)
    if metric=="lpips_edit_part":
        if src_mask.sum()==0 or tgt_mask.sum()==0:
            return "nan"
        else:
            return metrics_calculator.calculate_lpips(src_image, tgt_image, src_mask, tgt_mask)
    if metric=="mse_edit_part":
        if src_mask.sum()==0 or tgt_mask.sum()==0:
            return "nan"
        else:
            return metrics_calculator.calculate_mse(src_image, tgt_image, src_mask, tgt_mask)
    if metric=="ssim_edit_part":
        if src_mask.sum()==0 or tgt_mask.sum()==0:
            return "nan"
        else:
            return metrics_calculator.calculate_ssim(src_image, tgt_image, src_mask, tgt_mask)
    if metric=="structure_distance_edit_part":
        if src_mask.sum()==0 or tgt_mask.sum()==0:
            return "nan"
        else:
            return metrics_calculator.calculate_structure_distance(src_image, tgt_image, src_mask, tgt_mask)
    if metric=="clip_similarity_source_image":
        return metrics_calculator.calculate_clip_similarity(src_image, src_prompt,None)
    if metric=="clip_similarity_target_image":
        return metrics_calculator.calculate_clip_similarity(tgt_image, tgt_prompt,None)
    if metric=="clip_similarity_target_image_edit_part":
        if tgt_mask.sum()==0:
            return "nan"
        else:
            return metrics_calculator.calculate_clip_similarity(tgt_image, tgt_prompt,tgt_mask)
    if metric=="local_clip":
        return metrics_calculator.compute_local_clip(src_image, src_prompt, tgt_image, tgt_prompt)
    
all_tgt_image_folders={
    "your_method" : "directory_of_images",
    
    # For example:
    # "h_edit_P": "~/results_h_edit_P/",

}


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_mapping_file', type=str, default="./PIE_Bench_Data/mapping_file.json")
    parser.add_argument('--metrics',  nargs = '+', type=str, default=[
                                                         "structure_distance",
                                                         "psnr_unedit_part",
                                                         "lpips_unedit_part",
                                                         "mse_unedit_part",
                                                         "ssim_unedit_part",
                                                         "clip_similarity_source_image",
                                                         "clip_similarity_target_image",
                                                         "clip_similarity_target_image_edit_part",
                                                         "local_clip"
                                                        ])
    parser.add_argument('--src_image_folder', type=str, default="./PIE_Bench_Data/annotation_images")
    parser.add_argument('--tgt_methods', nargs = '+', type=str, default=[
                                                            "your_method",
                                                            ])
    
    parser.add_argument('--result_path', type=str, default="./results/results.csv")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--edit_category_list',  nargs = '+', type=str, default=[
                                                                                "0",
                                                                                "1",
                                                                                "2",
                                                                                "3",
                                                                                "4",
                                                                                "5",
                                                                                "6",
                                                                                "7",
                                                                                "8",
                                                                                "9"
                                                                                ]) # the editing category that needed to run
    parser.add_argument('--evaluate_whole_table', action= "store_true") # rerun existing images

    args = parser.parse_args()
    
    annotation_mapping_file=args.annotation_mapping_file
    metrics=args.metrics
    src_image_folder=args.src_image_folder
    tgt_methods=args.tgt_methods
    edit_category_list=args.edit_category_list
    evaluate_whole_table=args.evaluate_whole_table
    
    tgt_image_folders={}
    
    if evaluate_whole_table:
        for key in all_tgt_image_folders:
            if key[0] in tgt_methods:
                tgt_image_folders[key]=all_tgt_image_folders[key]
    else:
        for key in tgt_methods:
            tgt_image_folders[key]=all_tgt_image_folders[key]
    
    result_path=args.result_path
    
    metrics_calculator=MetricsCalculator(args.device)
    
    with open(result_path,'w',newline="") as f:
        csv_write = csv.writer(f)
        
        csv_head=[]
        for tgt_image_folder_key,_ in tgt_image_folders.items():
            for metric in metrics:
                csv_head.append(f"{tgt_image_folder_key}|{metric}")
        
        data_row = ["file_id"]+csv_head
        csv_write.writerow(data_row)

    with open(annotation_mapping_file,"r") as f:
        annotation_file=json.load(f)

    for key, item in annotation_file.items():
        if item["editing_type_id"] not in edit_category_list:
            continue

        print(f"evaluating image {key} ...")
        base_image_path=item["image_path"]
        mask=mask_decode(item["mask"])
        original_prompt = item["original_prompt"].replace("[", "").replace("]", "")
        editing_prompt = item["editing_prompt"].replace("[", "").replace("]", "")
        
        mask=mask[:,:,np.newaxis].repeat([3],axis=2)
        
        src_image_path=os.path.join(src_image_folder, base_image_path)
        src_image = Image.open(src_image_path)
        
        evaluation_result=[key]
        
        for tgt_image_folder_key,tgt_image_folder in tgt_image_folders.items():
            tgt_image_path=os.path.join(tgt_image_folder, base_image_path)
            print(f"evaluating method: {tgt_image_folder_key}")
            
            tgt_image = Image.open(tgt_image_path)
            if tgt_image.size[0] != tgt_image.size[1]:
                # to evaluate editing
                tgt_image = tgt_image.crop((tgt_image.size[0]-512,tgt_image.size[1]-512,tgt_image.size[0],tgt_image.size[1])) 
                # to evaluate reconstruction
                # tgt_image = tgt_image.crop((tgt_image.size[0]-512*2,tgt_image.size[1]-512,tgt_image.size[0]-512,tgt_image.size[1])) 
            
            for metric in metrics:
                print(f"evluating metric: {metric}")
                evaluation_result.append(calculate_metric(metrics_calculator,metric,src_image, tgt_image, mask, mask, original_prompt, editing_prompt))
                        
        with open(result_path,'a+',newline="") as f:
            csv_write = csv.writer(f)
            csv_write.writerow(evaluation_result)
    