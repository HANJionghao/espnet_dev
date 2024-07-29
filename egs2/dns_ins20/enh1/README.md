<!-- Generated by ./scripts/utils/show_enh_score.sh -->
# RESULTS
## Environments
- date: `Fri May 28 20:55:14 CST 2021`
- python version: `3.8.5 (default, Sep  4 2020, 07:30:14)  [GCC 7.3.0]`
- espnet version: `espnet 0.9.9`
- pytorch version: `pytorch 1.4.0`
- Git hash: `be73d0de071e9a7fcaf98ad2e5c94dad9ca73cda`
  - Commit date: `Fri May 28 19:14:45 2021 +0800`


## enh_train_enh_blstm_tf_raw

 - config: ./conf/tuning/train_enh_blstm_tf.yaml
 - Pretrained model: https://zenodo.org/record/4923697

| dataset                           | STOI | SAR   | SDR   | SIR  |
| --------------------------------- | ---- | ----- | ----- | ---- |
| enhanced_cv_synthetic             | 0.95 | 18.63 | 18.63 | 0.00 |
| enhanced_tt_synthetic_no_reverb   | 0.92 | 10.92 | 10.92 | 0.00 |
| enhanced_tt_synthetic_with_reverb | 0.85 | 9.31  | 9.31  | 0.00 |

<!-- Generated by ./scripts/utils/show_enh_score.sh -->
# RESULTS
## Environments
- date: `Thu Feb 10 23:11:40 CST 2022`
- python version: `3.8.12 (default, Oct 12 2021, 13:49:34)  [GCC 7.5.0]`
- espnet version: `espnet 0.10.5a1`
- pytorch version: `pytorch 1.9.1`
- Git hash: `6f66283b9eed7b0d5e5643feb18d8f60118a4afc`
  - Commit date: `Mon Dec 13 15:30:29 2021 +0800`


## enh_train_enh_dccrn_raw

- config: ./conf/tuning/train_enh_dccrn.yaml
- download_model: https://huggingface.co/Johnson-Lsx/Shaoxiong_Lin_dns_ins20_enh_enh_train_enh_dccrn_raw

| dataset                           | PESQ | STOI | SAR   | SDR   | SIR  | SI_SNR |
| --------------------------------- | ---- | ---- | ----- | ----- | ---- | ------ |
| enhanced_cv_synthetic             | 3.72 | 0.98 | 24.69 | 24.69 | 0.00 | 24.22  |
| enhanced_tt_synthetic_no_reverb   | 3.29 | 0.96 | 17.69 | 17.69 | 0.00 | 17.50  |
| enhanced_tt_synthetic_with_reverb | 2.54 | 0.81 | 10.45 | 10.45 | 0.00 | 9.72   |

Note: Here, the model is only trained on data without reverberation.
Note: Here, the PESQ score is calculated based on https://github.com/vBaiCai/python-pesq.


<!-- Generated by ./scripts/utils/show_enh_score.sh -->
# RESULTS
## Environments
- date: `Sun Apr 24 23:59:01 EDT 2022`
- python version: `3.8.12 (default, Oct 12 2021, 13:49:34)  [GCC 7.5.0]`
- espnet version: `espnet 0.10.6a1`
- pytorch version: `pytorch 1.10.2+cu102`
- Git hash: `21c02c8f578b9860e6bf38c86a0bd7cd0412c7f8`
  - Commit date: `Sun Feb 6 15:37:51 2022 -0500`


## enh_train_enh_conv_tasnet_raw

- config: ./conf/tuning/train_enh_conv_tasnet.yaml
- model: https://huggingface.co/muqiaoy/muqiaoy_dns_ins20_enh_train_enh_conv_tasnet_raw


| dataset                           | STOI | SAR   | SDR   | SI_SNR  |
| --------------------------------- | ---- | ----- | ----- | ---- |
| enhanced_cv_synthetic             | 0.97 | 24.52 | 24.52 | 24.43 |
| enhanced_tt_synthetic_no_reverb   | 0.96 | 17.66 | 17.66 | 17.69 |
| enhanced_tt_synthetic_with_reverb | 0.84 | 11.84 | 11.84 | 11.15 |


<!-- Generated by ./scripts/utils/show_enh_score.sh -->
# RESULTS
## Environments
- date: `Wed Mar  6 01:29:02 UTC 2024`
- python version: `3.10.10 (main, Mar 21 2023, 18:45:11) [GCC 11.2.0]`
- espnet version: `espnet 202308`
- pytorch version: `pytorch 2.1.0+cu118`
- Git hash: `60ce18efa06ca5a5922534682f47e2107ef88b13`
  - Commit date: `Wed Sep 6 10:17:57 2023 -0700`


## enh_train_enh_tfgrid_raw

- config: ./conf/tuning/train_enh_tfgrid.yaml
- model: https://huggingface.co/Zhaoheng/tfgridnet_dns_ins20_epoch33

|dataset|PESQ_WB|STOI|SAR|SDR|SIR|SI_SNR|
|---|---|---|---|---|---|---|
|enhanced_cv_synthetic|3.61|99.06|26.04|26.04|0.00|26.44|
|enhanced_tt_synthetic_no_reverb|3.32|97.88|20.18|20.18|0.00|20.17|
|enhanced_tt_synthetic_with_reverb|2.79|91.75|15.54|15.54|0.00|15.06|