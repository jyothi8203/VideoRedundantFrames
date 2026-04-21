import os
import json
rt_pth = 'D:/Project/Datasets/scene_text/total_text/test_gts'
gt_dict = {}
# Iterate through all files in the folder
for filename in os.listdir(rt_pth):
    if filename.endswith(".txt"):
        file_path = os.path.join(rt_pth, filename)

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        bboxes = []
        texts = []

        # Assuming format: xmin ymin xmax ymax text...
        for line in lines:
            # parts = line.strip().split(' ')
            parts = line.split(',')
            if len(parts) >= 5:
                bboxes.append([parts[:-1]])
                texts.append(" ".join(parts[-1]))

        # Add to dictionary
        data_dict[filename] = {
            "num_entities": len(bboxes),
            "bboxes": bboxes,
            "texts": texts
        }

# Write to JSON file
output_file = 'D:/Project/Datasets/scene_text/totaltext/Images/Train/gt_data.json'
with open(output_file, 'w', encoding='utf-8') as jf:
    json.dump(gt_dict, jf, indent=4)
