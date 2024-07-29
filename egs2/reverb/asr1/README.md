<!-- Generated by scripts/utils/show_asr_result.sh -->
# RESULTS
## Transformer ASR + Transformer  LM  + SpeedPerturbation + SpecAug + applying RIR and noise data on the fly
### Environments
- date: `Fri Jan 15 10:04:32 JST 2021`
- python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
- espnet version: `espnet 0.9.5`
- pytorch version: `pytorch 1.5.1`
- Git hash: `1bcf69d5d8c724cded6e5f9abef68e8000fd4b57`
  - Commit date: `Mon Jan 4 13:47:44 2021 +0900`

### Config
- ASR: [conf/tuning/train_asr_transformer4.yaml](conf/tuning/train_asr_transformer4.yaml)
- LM: [conf/tuning/train_lm_transformer.yaml](conf/tuning/train_lm_transformer.yaml)
- Decode: [conf/tuning/decode.yaml](conf/tuning/decode.yaml)
- Pretrained model: [https://zenodo.org/record/4441309/files/asr_train_asr_transformer2_raw_en_char_rir_scpdatareverb_rir_singlewav.scp_noise_db_range12_17_noise_scpdatareverb_noise_singlewav.scp_speech_volume_normalize1.0_num_workers2_rir_apply_prob0.999_noise_apply_prob1._sp_valid.acc.ave.zip?download=1](https://zenodo.org/record/4441309/files/asr_train_asr_transformer2_raw_en_char_rir_scpdatareverb_rir_singlewav.scp_noise_db_range12_17_noise_scpdatareverb_noise_singlewav.scp_speech_volume_normalize1.0_num_workers2_rir_apply_prob0.999_noise_apply_prob1._sp_valid.acc.ave.zip?download=1)

### No frontend
####  WER
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/dt_real_1ch_far|89|1463|93.2|5.3|1.4|1.2|7.9|49.4|
|decode_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/dt_real_1ch_near|90|1603|94.8|3.9|1.2|0.6|5.8|47.8|
|decode_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/dt_simu_1ch_far|742|12169|95.5|3.6|0.9|0.3|4.8|38.5|
|decode_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/dt_simu_1ch_near|742|12169|96.9|2.5|0.6|0.2|3.3|29.9|
|decode_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/et_real_1ch_far|186|2962|94.4|4.5|1.1|0.7|6.3|41.9|
|decode_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/et_real_1ch_near|186|3131|94.8|4.2|1.0|0.8|6.0|45.7|
|decode_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/et_simu_1ch_far|1088|17986|95.7|3.5|0.8|0.4|4.7|39.3|
|decode_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/et_simu_1ch_near|1088|17986|96.6|2.9|0.6|0.3|3.7|34.3|

#### CER
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/dt_real_1ch_far|89|8845|96.9|1.6|1.5|1.1|4.2|49.4|
|decode_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/dt_real_1ch_near|90|9336|97.8|1.1|1.1|0.9|3.1|47.8|
|decode_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/dt_simu_1ch_far|742|71524|98.1|1.0|0.9|0.4|2.3|38.9|
|decode_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/dt_simu_1ch_near|742|71524|98.8|0.6|0.6|0.3|1.5|30.3|
|decode_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/et_real_1ch_far|186|17261|97.5|1.3|1.2|0.9|3.4|41.9|
|decode_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/et_real_1ch_near|186|18433|97.9|1.1|1.0|0.9|3.0|45.7|
|decode_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/et_simu_1ch_far|1088|105480|98.3|0.9|0.9|0.4|2.2|40.1|
|decode_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/et_simu_1ch_near|1088|105480|98.7|0.7|0.7|0.3|1.7|35.1|

### 1ch WPE
####  WER
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/dt_real_1ch_wpe_far|89|1463|93.3|5.5|1.2|1.2|7.9|48.3|
|decode_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/dt_real_1ch_wpe_near|90|1603|95.6|3.4|0.9|0.7|5.1|44.4|
|decode_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/dt_simu_1ch_wpe_far|742|12169|95.7|3.5|0.8|0.3|4.6|37.9|
|decode_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/dt_simu_1ch_wpe_near|742|12169|96.9|2.6|0.6|0.2|3.4|30.3|
|decode_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/et_real_1ch_wpe_far|186|2962|94.9|4.1|1.1|0.6|5.8|39.2|
|decode_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/et_real_1ch_wpe_near|186|3131|95.3|3.9|0.8|0.7|5.5|43.0|
|decode_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/et_simu_1ch_wpe_far|1088|17986|95.8|3.5|0.8|0.3|4.6|39.1|
|decode_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/et_simu_1ch_wpe_near|1088|17986|96.6|2.8|0.6|0.3|3.7|34.8|

#### CER
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/dt_real_1ch_wpe_far|89|8845|97.1|1.5|1.4|1.1|4.0|48.3|
|decode_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/dt_real_1ch_wpe_near|90|9336|98.1|0.9|0.9|0.8|2.7|44.4|
|decode_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/dt_simu_1ch_wpe_far|742|71524|98.2|1.0|0.9|0.4|2.2|38.3|
|decode_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/dt_simu_1ch_wpe_near|742|71524|98.8|0.6|0.6|0.3|1.5|30.7|
|decode_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/et_real_1ch_wpe_far|186|17261|97.8|1.2|1.0|0.9|3.1|39.2|
|decode_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/et_real_1ch_wpe_near|186|18433|98.0|1.1|0.9|0.9|2.8|43.0|
|decode_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/et_simu_1ch_wpe_far|1088|105480|98.3|0.8|0.9|0.4|2.1|39.8|
|decode_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/et_simu_1ch_wpe_near|1088|105480|98.7|0.6|0.7|0.3|1.6|35.7|

### 8ch WPE+Beamformit
####  WER
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/dt_real_8ch_beamformit_far|89|1463|95.6|3.7|0.8|0.7|5.1|42.7|
|decode_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/dt_real_8ch_beamformit_near|90|1603|96.6|2.7|0.6|0.4|3.8|38.9|
|decode_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/dt_simu_8ch_beamformit_far|742|12169|96.8|2.6|0.6|0.3|3.5|30.5|
|decode_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/dt_simu_8ch_beamformit_near|742|12169|97.1|2.3|0.6|0.2|3.1|29.4|
|decode_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/et_real_8ch_beamformit_far|186|2962|96.2|3.3|0.5|0.6|4.4|34.4|
|decode_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/et_real_8ch_beamformit_near|186|3131|96.8|2.7|0.5|0.4|3.6|32.8|
|decode_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/et_simu_8ch_beamformit_far|1088|17986|96.7|2.8|0.5|0.4|3.7|33.8|
|decode_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/et_simu_8ch_beamformit_near|1088|17986|96.8|2.7|0.5|0.3|3.5|33.0|

#### CER
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/dt_real_8ch_beamformit_far|89|8845|98.2|0.9|0.9|0.5|2.4|42.7|
|decode_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/dt_real_8ch_beamformit_near|90|9336|98.6|0.7|0.7|0.5|1.9|38.9|
|decode_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/dt_simu_8ch_beamformit_far|742|71524|98.7|0.6|0.6|0.3|1.6|30.9|
|decode_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/dt_simu_8ch_beamformit_near|742|71524|98.9|0.6|0.6|0.2|1.4|29.8|
|decode_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/et_real_8ch_beamformit_far|186|17261|98.5|0.8|0.7|0.6|2.0|34.4|
|decode_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/et_real_8ch_beamformit_near|186|18433|98.7|0.7|0.6|0.5|1.8|32.8|
|decode_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/et_simu_8ch_beamformit_far|1088|105480|98.7|0.7|0.6|0.4|1.7|34.2|
|decode_lm_lm_train_lm_transformer_en_char_valid.loss.ave_asr_model_valid.acc.ave/et_simu_8ch_beamformit_near|1088|105480|98.8|0.6|0.6|0.3|1.6|33.6|

## Transformer + SpeedPerturbation + SpecAug
### Environments
 - date: `Wed Nov 18 08:43:05 JST 2020`
 - python version: `3.7.3 (default, Mar 27 2019, 22:11:17)  [GCC 7.3.0]`
 - espnet version: `espnet 0.9.5`
 - pytorch version: `pytorch 1.5.1`
 - Git hash: `7aad8240a2c0643289ce6ea76d6f42eb12c15674`
   - Commit date: `Tue Nov 10 22:46:38 2020 +0900`
- ASR config: [conf/tuning/train_asr_transformer.yaml](conf/tuning/train_asr_transformer.yaml)
- LM config: [conf/tuning/train_lm_transformer.yaml](conf/tuning/train_lm_transformer.yaml)
- Decode config: [conf/tuning/decode.yaml](conf/tuning/decode.yaml)
- Pretrained model: https://zenodo.org/record/4278363

### No frontend
#### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_lm_lm_train_lm_transformer_char_valid.loss.ave_asr_model_valid.acc.ave/dt_real_1ch_far|89|1463|90.8|7.4|1.8|1.4|10.5|60.7|
|decode_lm_lm_train_lm_transformer_char_valid.loss.ave_asr_model_valid.acc.ave/dt_real_1ch_near|90|1603|92.0|6.0|2.1|0.6|8.6|50.0|
|decode_lm_lm_train_lm_transformer_char_valid.loss.ave_asr_model_valid.acc.ave/dt_simu_1ch_far|742|12169|95.5|3.5|1.0|0.4|4.9|38.9|
|decode_lm_lm_train_lm_transformer_char_valid.loss.ave_asr_model_valid.acc.ave/dt_simu_1ch_near|742|12169|97.0|2.4|0.6|0.2|3.2|28.8|
|decode_lm_lm_train_lm_transformer_char_valid.loss.ave_asr_model_valid.acc.ave/et_real_1ch_far|186|2962|90.7|7.5|1.8|1.3|10.5|54.8|
|decode_lm_lm_train_lm_transformer_char_valid.loss.ave_asr_model_valid.acc.ave/et_real_1ch_near|186|3131|92.4|6.3|1.3|0.8|8.3|51.1|
|decode_lm_lm_train_lm_transformer_char_valid.loss.ave_asr_model_valid.acc.ave/et_simu_1ch_far|1088|17986|95.7|3.6|0.8|0.4|4.7|40.2|
|decode_lm_lm_train_lm_transformer_char_valid.loss.ave_asr_model_valid.acc.ave/et_simu_1ch_near|1088|17986|97.2|2.3|0.6|0.3|3.1|31.3|

#### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_lm_lm_train_lm_transformer_char_valid.loss.ave_asr_model_valid.acc.ave/dt_real_1ch_far|89|8845|95.5|2.2|2.2|1.2|5.7|60.7|
|decode_lm_lm_train_lm_transformer_char_valid.loss.ave_asr_model_valid.acc.ave/dt_real_1ch_near|90|9336|96.1|1.8|2.1|0.9|4.8|50.0|
|decode_lm_lm_train_lm_transformer_char_valid.loss.ave_asr_model_valid.acc.ave/dt_simu_1ch_far|742|71524|98.1|0.9|1.0|0.4|2.3|39.2|
|decode_lm_lm_train_lm_transformer_char_valid.loss.ave_asr_model_valid.acc.ave/dt_simu_1ch_near|742|71524|98.8|0.5|0.7|0.2|1.4|29.1|
|decode_lm_lm_train_lm_transformer_char_valid.loss.ave_asr_model_valid.acc.ave/et_real_1ch_far|186|17261|95.5|2.3|2.2|1.3|5.8|54.8|
|decode_lm_lm_train_lm_transformer_char_valid.loss.ave_asr_model_valid.acc.ave/et_real_1ch_near|186|18433|96.5|1.9|1.6|0.8|4.3|51.1|
|decode_lm_lm_train_lm_transformer_char_valid.loss.ave_asr_model_valid.acc.ave/et_simu_1ch_far|1088|105480|98.2|0.9|0.9|0.4|2.2|40.8|
|decode_lm_lm_train_lm_transformer_char_valid.loss.ave_asr_model_valid.acc.ave/et_simu_1ch_near|1088|105480|98.9|0.5|0.6|0.2|1.3|32.0|

### 1ch WPE
#### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_lm_lm_train_lm_transformer_char_valid.loss.ave_asr_model_valid.acc.ave/dt_real_1ch_wpe_far|89|1463|91.4|6.8|1.8|1.1|9.7|58.4|
|decode_lm_lm_train_lm_transformer_char_valid.loss.ave_asr_model_valid.acc.ave/dt_real_1ch_wpe_near|90|1603|92.2|5.7|2.1|0.6|8.4|50.0|
|decode_lm_lm_train_lm_transformer_char_valid.loss.ave_asr_model_valid.acc.ave/dt_simu_1ch_wpe_far|742|12169|95.7|3.4|0.9|0.4|4.8|38.0|
|decode_lm_lm_train_lm_transformer_char_valid.loss.ave_asr_model_valid.acc.ave/dt_simu_1ch_wpe_near|742|12169|97.0|2.3|0.6|0.2|3.2|28.8|
|decode_lm_lm_train_lm_transformer_char_valid.loss.ave_asr_model_valid.acc.ave/et_real_1ch_wpe_far|186|2962|91.4|6.8|1.8|1.1|9.8|54.8|
|decode_lm_lm_train_lm_transformer_char_valid.loss.ave_asr_model_valid.acc.ave/et_real_1ch_wpe_near|186|3131|93.1|5.7|1.2|0.8|7.7|52.7|
|decode_lm_lm_train_lm_transformer_char_valid.loss.ave_asr_model_valid.acc.ave/et_simu_1ch_wpe_far|1088|17986|95.9|3.4|0.7|0.3|4.5|39.2|
|decode_lm_lm_train_lm_transformer_char_valid.loss.ave_asr_model_valid.acc.ave/et_simu_1ch_wpe_near|1088|17986|97.2|2.3|0.5|0.2|3.0|31.1|

#### CER
|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_lm_lm_train_lm_transformer_char_valid.loss.ave_asr_model_valid.acc.ave/dt_real_1ch_wpe_far|89|8845|95.9|1.9|2.2|1.2|5.3|58.4|
|decode_lm_lm_train_lm_transformer_char_valid.loss.ave_asr_model_valid.acc.ave/dt_real_1ch_wpe_near|90|9336|96.3|1.8|1.9|0.8|4.5|50.0|
|decode_lm_lm_train_lm_transformer_char_valid.loss.ave_asr_model_valid.acc.ave/dt_simu_1ch_wpe_far|742|71524|98.1|0.9|1.0|0.4|2.3|38.3|
|decode_lm_lm_train_lm_transformer_char_valid.loss.ave_asr_model_valid.acc.ave/dt_simu_1ch_wpe_near|742|71524|98.8|0.5|0.7|0.2|1.4|29.1|
|decode_lm_lm_train_lm_transformer_char_valid.loss.ave_asr_model_valid.acc.ave/et_real_1ch_wpe_far|186|17261|95.8|2.2|2.0|1.1|5.3|54.8|
|decode_lm_lm_train_lm_transformer_char_valid.loss.ave_asr_model_valid.acc.ave/et_real_1ch_wpe_near|186|18433|97.0|1.7|1.3|0.8|3.8|52.7|
|decode_lm_lm_train_lm_transformer_char_valid.loss.ave_asr_model_valid.acc.ave/et_simu_1ch_wpe_far|1088|105480|98.4|0.8|0.9|0.4|2.0|39.8|
|decode_lm_lm_train_lm_transformer_char_valid.loss.ave_asr_model_valid.acc.ave/et_simu_1ch_wpe_near|1088|105480|98.9|0.5|0.6|0.2|1.3|31.9|


### 8ch WPE+Beamformit
#### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_lm_lm_train_lm_transformer_char_valid.loss.ave_asr_model_valid.acc.ave/dt_real_8ch_beamformit_far|89|1463|95.0|4.0|1.0|0.7|5.7|38.2|
|decode_lm_lm_train_lm_transformer_char_valid.loss.ave_asr_model_valid.acc.ave/dt_real_8ch_beamformit_near|90|1603|96.3|2.8|0.9|0.4|4.1|33.3|
|decode_lm_lm_train_lm_transformer_char_valid.loss.ave_asr_model_valid.acc.ave/dt_simu_8ch_beamformit_far|742|12169|97.2|2.2|0.6|0.2|2.9|27.5|
|decode_lm_lm_train_lm_transformer_char_valid.loss.ave_asr_model_valid.acc.ave/dt_simu_8ch_beamformit_near|742|12169|97.3|2.1|0.6|0.2|2.8|25.7|
|decode_lm_lm_train_lm_transformer_char_valid.loss.ave_asr_model_valid.acc.ave/et_real_8ch_beamformit_far|186|2962|96.0|3.4|0.6|0.7|4.7|33.9|
|decode_lm_lm_train_lm_transformer_char_valid.loss.ave_asr_model_valid.acc.ave/et_real_8ch_beamformit_near|186|3131|96.5|2.6|0.9|0.5|4.0|33.3|
|decode_lm_lm_train_lm_transformer_char_valid.loss.ave_asr_model_valid.acc.ave/et_simu_8ch_beamformit_far|1088|17986|97.4|2.1|0.5|0.2|2.9|29.5|
|decode_lm_lm_train_lm_transformer_char_valid.loss.ave_asr_model_valid.acc.ave/et_simu_8ch_beamformit_near|1088|17986|97.5|2.0|0.5|0.2|2.7|28.6|

#### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_lm_lm_train_lm_transformer_char_valid.loss.ave_asr_model_valid.acc.ave/dt_real_8ch_beamformit_far|89|8845|98.0|1.0|1.1|0.7|2.8|38.2|
|decode_lm_lm_train_lm_transformer_char_valid.loss.ave_asr_model_valid.acc.ave/dt_real_8ch_beamformit_near|90|9336|98.6|0.6|0.8|0.5|1.9|33.3|
|decode_lm_lm_train_lm_transformer_char_valid.loss.ave_asr_model_valid.acc.ave/dt_simu_8ch_beamformit_far|742|71524|99.0|0.4|0.6|0.2|1.2|27.9|
|decode_lm_lm_train_lm_transformer_char_valid.loss.ave_asr_model_valid.acc.ave/dt_simu_8ch_beamformit_near|742|71524|99.0|0.4|0.6|0.2|1.2|26.1|
|decode_lm_lm_train_lm_transformer_char_valid.loss.ave_asr_model_valid.acc.ave/et_real_8ch_beamformit_far|186|17261|98.4|0.9|0.7|0.8|2.4|33.9|
|decode_lm_lm_train_lm_transformer_char_valid.loss.ave_asr_model_valid.acc.ave/et_real_8ch_beamformit_near|186|18433|98.6|0.6|0.8|0.6|1.9|33.3|
|decode_lm_lm_train_lm_transformer_char_valid.loss.ave_asr_model_valid.acc.ave/et_simu_8ch_beamformit_far|1088|105480|99.0|0.4|0.6|0.2|1.2|30.1|
|decode_lm_lm_train_lm_transformer_char_valid.loss.ave_asr_model_valid.acc.ave/et_simu_8ch_beamformit_near|1088|105480|99.0|0.4|0.6|0.2|1.2|29.6|