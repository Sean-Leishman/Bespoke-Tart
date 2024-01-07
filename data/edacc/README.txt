# The Edinburgh International Accents of English Corpus V1.0

Thank you for choosing The Edinburgh International Accents of English Corpus. In this document, we describe all the files included in the V1.0 release. For more information, visit https://groups.inf.ed.ac.uk/edacc, and for comments or questions, please contact us at ramon.sanabria.teixidor@gmail.com.

#Files

- In ./dev and ./test, you will find all files needed to evaluate each dataset. Specifically, in each folder you will find:
        - conv.list: a list of conversations included in this set
        - company.ctm: transcriptions from a well-known (anonymized) company engine in CTM format
        - text: textual transcription of each turn
        - segment: time alignment for each transcription and the audio file name
        - utt2spk: corresponding speaker for each segment
        - stm: transcription file needed for evaluation (see #Evaluation Section)

- ./data contains the raw recordings of each conversation

- ./linguistic_background.csv contains the linguistic background reported by every speaker

- If you are interested in working only with read speech, you can find them labeled as IGNORE_TIME_SEGMENT_IN_SCORING in the STM file

# Special Tokens

- We use some special tokens to designate non-transcribable tokens. These are consistent with: https://github.com/kaldi-asr/kaldi/blob/master/egs/babel/s5d/local/prepare_acoustic_training_data.pl#L68-L89
        - <breath>
        - <click>
        - <cough>
        - <dtmf>
        - <foreign>
        - <laugh>
        - <lipsmack>
        - <no-speech>
        - <overlap>
        - <ring>

- We mark read speech segments in the STM files as IGNORE_TIME_SEGMENT_IN_SCORING

#Evaluation

- ./evaluate.sh is a script that computes WER using sclite on the development and test sets
    - Set KALDI_ROOT with your Kaldi directory
    - It uses hypotheses in CTM format
    - It uses references in STM format
    - It uses the ./glm file to:
        - Normalize transcriptions (e.g., hm -> hmm)
        - Remove para-linguistic sounds such as hesitations and noises during scoring
     - We have incorporated speaker category labels in the STM file as examples to demonstrate how to report performance on distinct speaker groups. These labels may not be entirely reliable (e.g., L1/L2), so please do not use them for reporting on these segments without verifying them. Instead, use these labels as a guide to understand how to configure them within our testing framework.

## IMPORTANT NOTE:

- We only consider linguistic elements during scoring, and remove:
        - Control sentence passage
        - Hesitation
        - Foreign words
        - Cross-talk
        - Noises
        - <breath>
        - <click>
        - <cough>
        - <dtmf>
        - <foreign>
        - <laugh>
        - <lipsmack>
        - <no-speech>
        - <overlap>
        - <ring>

- If you want to consider any of the previous elements or evaluate them differently, you would need to modify the evaluation script/configuration files

# Citation

If you use our work, we would highly appreciate it if you could include a citation to EdACC. We provide it below:

@inproceedings{sanabria23edacc,
   title={{T}he {E}dinburgh {I}nternational {A}ccents of {E}nglish {C}orpus: {T}owards the {D}emocratization of {E}nglish {ASR}},
   author={Sanabria, Ramon and Bogoychev, Nikolay and  Markl, Nina and Carmantini, Andrea and  Klejch, Ondrej and Bell, Peter},
   booktitle={ICASSP},
   year={2023},
}
