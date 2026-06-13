DATASET=/mnt/localssd/calvin/
VIDEO=/mnt/localssd/calvin/weights/svd-robot-calvin-ft/
CLIP=/mnt/localssd/calvin/weights/clip-vit-base-patch32/
accelerate launch step2_train_action_calvin_hiva.py --root_data_dir ${DATASET} --video_model_path ${VIDEO} --text_encoder_path ${CLIP}