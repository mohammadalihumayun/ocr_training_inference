# modified from utrnet by Abdulrahman et. al. disabling data augmentation.
#! git clone https://github.com/mohammadalihumayun/UTRNet-High-Resolution-Urdu-Text-Recognition.git
#! pip install lmdb
#! pip install timm==0.6.12
# %cd '/content/UTRNet-High-Resolution-Urdu-Text-Recognition/'


# training from scratch
#! python3 train.py --train_data /content/val --valid_data /content/val --freeze_visual 'no' --FeatureExtraction HRNet --SequenceModeling DBiLSTM --Prediction CTC --exp_name UTRNet-Large --num_epochs 2 --batch_size 2

# fine tuning from first trained model (set freeze visual to 'yes' or 'no' to freeze cnn based visual feature extraction)
#! python3 train.py --train_data /content/train --valid_data /content/val --saved_model 'path/z1_e2e_best_norm_ED.pth' --freeze_visual 'no' --Prediction CTC --exp_name UTRNet-Large --num_epochs 1 --batch_size 6

# inference to save results reading from a source folder

# change option to images_path for modified read.py to read and compute cer for multiple lines
#! python3 read_save.py --output_text /content/drive/MyDrive/ocr_urdu/seerat/preds.csv --images_path source_rois_path/rois --FeatureExtraction HRNet --SequenceModeling DBiLSTM --Prediction CTC  --saved_model  path/best_norm_ED_tuned_e2e_p14_p70_ep3.pth
