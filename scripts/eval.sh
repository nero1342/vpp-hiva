DATASET=/mnt/localssd/calvin/
VIDEO=/mnt/localssd/calvin/weights/svd-robot-calvin-ft/
CLIP=/mnt/localssd/calvin/weights/clip-vit-base-patch32/
ACTION=/mnt/localssd/calvin/weights/dp-calvin
accelerate launch --num_processes=8 policy_evaluation/calvin_evaluate_multi.py --video_model_path ${VIDEO} --action_model_folder ${ACTION} --clip_model_path ${CLIP} --calvin_abc_dir ${DATASET} 
