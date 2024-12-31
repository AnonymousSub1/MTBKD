**This repository contains code for the paper "Multi-Teacher Bayesian Knowledge Distillation"**

We provide the code for the proposed MT-BKD algorithm.

In the following, we provide the reproducibility workflow for the Deeploc2 dataset used in the paper.


### Tutorial

**Step 1: pre-train the teacher models and get the predicted probability**

To release the computational burden for the readers, we provide the predicted probabilities of the pre-trained teacher models directly in [teacher_pred_prob](https://github.com/AnonymousSub1/MTBKD/tree/main/teacher_pred_prob).

**Step 2: distill the student model through MT-BKD**

The following code achieves the distillation of the student model through MT-BKD. The student models will be stored in './student_model_distill_mix_weighted/' and the MC sample of student model parameters will be stroed in './LMC_params_mix_weighted/'.

```
nohup python ESM_8M_distill_LMC_mix_weighted.py --epoch 200 --traindata_dir "./data/data_our_train_sample.csv" --batch_size 64 --student_dir './student_model_distill_mix_weighted/' --LMC_params './LMC_params_mix_weighted/' --path_plot './plot_student/'  > ./log_LMC/ESM_8M_LMC_mix_weighted_epoch200.log < /dev/null &
```

**Step 3: get inference and evaluate**

The empirical distribution of predicted probabilities of the student model can be achieved through:

```
nohup python ESM_8M_inference_p.py --epoch 200 --batch_size 64 --pred_dir './result_pred_p_mix_weighted' --LMC_params './LMC_params_mix_weighted/' > ./log_pred/ESM_8M_inference_epoch200_mix_weighte.log < /dev/null &
```

for multiple label data we used:

nohup python ESM_8M_inference_p.py --epoch 200 --batch_size 64 --testdata_dir "./data/data_MuLabel_sub.csv" --pred_dir './result_pred_p_mix_weighted_Mul' --LMC_params './LMC_params_mix_weighted/' > ./log_pred/ESM_8M_inference_epoch200_mix_weighte_Mul.log < /dev/null &


The evaluation is available in [pred_deviance_infer_git.ipynb](https://github.com/AnonymousSub1/MTBKD/blob/main/pred_deviance_infer_git.ipynb)




