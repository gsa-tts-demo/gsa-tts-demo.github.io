from typing import List, Tuple, Union
from tqdm import tqdm
from glob import glob
import os
import whisper
import editdistance
import jiwer
import argparse
import json
from collections import defaultdict
from whisper.normalizers import EnglishTextNormalizer


def word_error_rate(hypotheses: List[str], references: List[str], use_cer=False) -> float:
    """
    Computes Average Word Error rate between two texts represented as
    corresponding lists of string.

    Hypotheses and references must have same length.

    Args:
        hypotheses (list): list of hypotheses
        references(list) : list of references
        use_cer (bool): set True to enable cer

    Returns:
        wer (float): average word error rate
    """
    scores = 0
    words = 0
    if len(hypotheses) != len(references):
        raise ValueError(
            "In word error rate calculation, hypotheses and reference"
            " lists must have the same number of elements. But I got:"
            "{0} and {1} correspondingly".format(len(hypotheses), len(references))
        )
    for h, r in zip(hypotheses, references):
        if use_cer:
            h_list = list(h)
            r_list = list(r)
        else:
            h_list = h.split()
            r_list = r.split()
        words += len(r_list)
        # May deprecate using editdistance in future release for here and rest of codebase
        # once we confirm jiwer is reliable.
        scores += editdistance.eval(h_list, r_list)
    if words != 0:
        wer = 1.0 * scores / words
    else:
        wer = float('inf')
    return wer

def word_error_rate_per_utt(hypotheses: List[str], references: List[str], use_cer=False) -> Tuple[List[float], float]:
    """
    Computes Word Error Rate per utterance and the average WER
    between two texts represented as corresponding lists of string. 
    
    Hypotheses and references must have same length.

    Args:
        hypotheses (list): list of hypotheses
        references(list) : list of references
        use_cer (bool): set True to enable cer

    Returns:
        wer_per_utt (List[float]): word error rate per utterance
        avg_wer (float): average word error rate
    """
    scores = 0
    words = 0
    wer_per_utt = []

    if len(hypotheses) != len(references):
        raise ValueError(
            "In word error rate calculation, hypotheses and reference"
            " lists must have the same number of elements. But I got:"
            "{0} and {1} correspondingly".format(len(hypotheses), len(references))
        )

    for h, r in zip(hypotheses, references):
        if use_cer:
            h_list = list(h)
            r_list = list(r)
        else:
            h_list = h.split()
            r_list = r.split()

        # To get rid of the issue that jiwer does not allow empty string
        if len(r_list) == 0:
            if len(h_list) != 0:
                errors = len(h_list)
                wer_per_utt.append(float('inf'))
        else:
            if use_cer:
                measures = jiwer.cer(r, h, return_dict=True)
                er = measures['cer']
            else:
                measures = jiwer.compute_measures(r, h)
                er = measures['wer']

            errors = measures['insertions'] + measures['deletions'] + measures['substitutions']
            wer_per_utt.append(er)

        scores += errors
        words += len(r_list)

    if words != 0:
        avg_wer = 1.0 * scores / words
    else:
        avg_wer = float('inf')

    return wer_per_utt, avg_wer


'''
meta_path로 어떤 모델 결과 계산할지 컨트롤
'''
# load whisper and normalizer
model = whisper.load_model("base.en")
normalizer = EnglishTextNormalizer()
model_name = 'GSA_residual'
_tag = 'Audio_control_only_noun_styleseg'
meta_path = f'infer/{model_name}/metas/all_test_metas_{model_name}.txt'
with open(meta_path, 'r') as fin:
    data = fin.read().splitlines()

hypotheses = []
references = []
for i, line in tqdm(enumerate(data[:654]), desc=f'Processing {model_name}...'):
    gt_path, speaker, gt_text, infer_path = line.strip().split('|')
    infer_path = infer_path.replace('Audio',_tag )
    # trans_result = model.transcribe(os.path.join(infer_path))
    try:
        trans_result = model.transcribe(infer_path)
        trans_text = trans_result["text"]

        gt_text_clean = normalizer(gt_text)
        trans_text_clean = normalizer(trans_text)
    
        references.append(gt_text_clean)
        hypotheses.append(trans_text_clean)
    except:
        continue

cer_per_utts, avg_cer = word_error_rate_per_utt(hypotheses, references, use_cer=True)
wer_per_utts, avg_wer = word_error_rate_per_utt(hypotheses, references, use_cer=False)

with open(f'CER_WER_{model_name}_{_tag}.txt', 'w') as fout_metric:
    fout_metric.writelines('index|CER|WER|Avg.CER|Avg.WER\n')
    for i, (cer_per_utt, wer_per_utt) in tqdm(enumerate(zip(cer_per_utts, wer_per_utts))):
        fout_metric.writelines(f'{i+1}|{cer_per_utt:.4f}|{wer_per_utt:.4f}|{avg_cer:.4f}|{avg_wer:.4f}\n')