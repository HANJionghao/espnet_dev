#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

# spectrogram-related arguments
fs=44100
if [ ${fs} -eq 24000 ];then
    fmin=0
    fmax=7600
    n_fft=2048
    n_shift=300
    win_length=1200
elif [ ${fs} -eq 44100 ]; then
    fmin=80
    fmax=22050
    n_fft=2048
    n_shift=512
    win_length=2048
fi

score_feats_extract=syllable_score_feats   # frame_score_feats | syllable_score_feats

opts="--audio_format wav "

train_set=tr_no_dev
valid_set=dev
test_sets="dev eval"

# training and inference configuration
train_config=conf/train.yaml
inference_config=conf/decode.yaml

# text related processing arguments
g2p=None
cleaner=none

pitch_extract=dio
ying_extract=None

combine_path=""
combine_path+="$(realpath ../../opencpop/svs1/dump/raw/)"
# combine_path+="\$$(realpath ../../opencpop/svs1/dump/raw/)"
# combine_path+="\$$(realpath ../../acesinger/svs1/dump/raw/)"
# combine_path+="\$$(realpath ../../kising/svs1/dump/raw/)"
# combine_path+="\$$(realpath ../../m4singer/svs1/dump/raw/)"
combine_path+="\$$(realpath ../../ameboshi/svs1/dump/raw/)"
combine_path+="\$$(realpath ../../kiritan/svs1/dump/raw/)"
# combine_path+="\$$(realpath ../../oniku_kurumi_utagoe_db/svs1/dump/raw/)"
# combine_path+="\$$(realpath ../../ofuton_p_utagoe_db/svs1/dump/raw/)"
# combine_path+="\$$(realpath ../../namine_ritsu_utagoe_db/svs1/dump/raw/)"
# combine_path+="\$$(realpath ../../itako/svs1/dump/raw/)"

datasets_to_fix_spk="kising" # more like "kising xxx yyy"

for dataset_name in ${datasets_to_fix_spk}; do
    kaldi_path=../../${dataset_name}/svs1/dump/raw
    ./local/update_spks_in_data_dir.sh ${kaldi_path} ${dataset_name}
done

use_sid=true
use_lid=false

min_wav_duration=2.0
gpu_inference=true

./svs.sh \
    --lang zh \
    --svs_task gan_svs \
    --local_data_opts "--combine_path ${combine_path} --stage 1" \
    --feats_type raw \
    --use_sid ${use_sid} \
    --use_lid ${use_lid} \
    --pitch_extract "${pitch_extract}" \
    --ying_extract "${ying_extract}" \
    --fs "${fs}" \
    --fmax "${fmax}" \
    --fmin "${fmin}" \
    --n_fft "${n_fft}" \
    --n_shift "${n_shift}" \
    --win_length "${win_length}" \
    --token_type phn \
    --g2p ${g2p} \
    --cleaner ${cleaner} \
    --min_wav_duration ${min_wav_duration} \
    --train_config "${train_config}" \
    --gpu_inference "${gpu_inference}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --score_feats_extract "${score_feats_extract}" \
    --srctexts "data/${train_set}/text" \
    --write_collected_feats true \
    ${opts} "$@"