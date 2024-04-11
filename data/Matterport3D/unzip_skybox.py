import os
from glob import glob
from tqdm import tqdm


input_dir = 'Matterport3D/v1/scans'
output_dir = 'mp3d_skybox'

zip_files = glob(os.path.join(input_dir, '*', 'matterport_skybox_images.zip'))
for zip_file in tqdm(zip_files):
    cmd = f"unzip -o {zip_file} -d {output_dir}"
    print(cmd)
    os.system(cmd)
