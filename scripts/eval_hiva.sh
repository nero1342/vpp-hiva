DATASET=/mnt/localssd/calvin/
VIDEO=/mnt/localssd/calvin/weights/svd-robot-calvin-ft/
CLIP=/mnt/localssd/calvin/weights/clip-vit-base-patch32/
ACTION=/home/colligo/Codes/HiVA/video-prediction-policy/logs/svd_224token_3dresampler_extraview_svd2_hiva/2026-06-12-09-38-38/checkpoints
# /home/colligo/Codes/HiVA/video-prediction-policy/logs/svd_224token_3dresampler_extraview_svd2_hiva/2026-06-12-06-30-55/checkpoints
# /home/colligo/Codes/HiVA/video-prediction-policy/logs/svd_224token_3dresampler_extraview_svd2_hiva/2026-06-12-06-43-40/checkpoints
# /home/colligo/Codes/HiVA/video-prediction-policy/cvpr2025/checkpoints
accelerate launch --num_processes=8 policy_evaluation/calvin_evaluate_hiva.py --video_model_path ${VIDEO} --action_model_folder ${ACTION} --clip_model_path ${CLIP} --calvin_abc_dir ${DATASET} 
