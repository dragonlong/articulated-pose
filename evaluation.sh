cd evaluation
python compute_gt_pose.py --item='eyeglasses' --domain='unseen' --nocs='ANCSH' --save

# run our processing over test group
python pose_multi_process.py --item='eyeglasses' --domain='unseen'

# pose & relative joint rotation
python eval_pose_err.py --item='eyeglasses' --domain='unseen' --nocs='ANCSH'

# 3d miou estimation
python compute_miou.py --item='eyeglasses' --domain='unseen' --nocs='ANCSH'

# performance on joint estimations 
python eval_joint_params.py --item='eyeglasses' --domain='unseen' --nocs='ANCSH'