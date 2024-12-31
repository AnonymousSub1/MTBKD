This is 


nohup python ESM_8M_distill_LMC_mix_weighted.py --epoch 200 --traindata_dir "./data/data_our_train_sample.csv" --batch_size 64 --student_dir './student_model_distill_mix_weighted/' --LMC_params './LMC_params_mix_weighted/' --path_plot './plot_student/'  > ./log_LMC/ESM_8M_LMC_mix_weighted_epoch200.log < /dev/null &

nohup python ESM_8M_inference_p.py --epoch 200 --batch_size 64 --pred_dir './result_pred_p_mix_weighted' --LMC_params './LMC_params_mix_weighted/' > ./log_pred/ESM_8M_inference_epoch200_mix_weighte.log < /dev/null &

