listPath='../../../data/Matterport3DLayoutAnnotation/data_list';
dataPath='../../../data/Matterport3D/mp3d_skybox';
out_dir='../../../data/Matterport3D/mp3d_skybox';

listFiles = {'mp3d_val.txt', 'mp3d_train.txt', 'mp3d_test.txt'};

for listFile = listFiles
  fprintf('Processing %s\n', listFile{1});
  listText=fileread(fullfile(listPath, listFile{1}));
  list=strsplit(listText, '\n');
  list=list(1, 1:end-1);

  if exist(out_dir,'dir')==0
        mkdir(out_dir);
  end
      
  i=0;
  for a=list(1:end)
      i=i+1;
      b=deblank(a{1});
      raw=strsplit(b);
      house_id=raw{1};
      image_id=raw{2};
      fprintf('house_id:%s image_id:%s\n',house_id, image_id);
      
      panorama_iamges_dir=fullfile(dataPath, house_id, 'matterport_stitched_images');
      panorama_iamge_path=fullfile(panorama_iamges_dir, sprintf('%s.png',image_id));
      output_iamge_folder=fullfile(out_dir, house_id, 'matterport_aligned_images');
      if ~exist(output_iamge_folder,'dir') 
          mkdir(output_iamge_folder);
      end
      output_iamge_path=fullfile(output_iamge_folder, sprintf('%s.png',image_id));
      
      if exist(panorama_iamge_path,'file')==0
        fprintf('%s not exist\n',panorama_iamge_path);
        continue;
      end
      
      if exist(output_iamge_path, 'file')==0
        preprocess(panorama_iamge_path, output_iamge_path);
      else
        disp('pass!');
        continue;
      end
      
      fprintf("----%d\n", i);
  end
end
