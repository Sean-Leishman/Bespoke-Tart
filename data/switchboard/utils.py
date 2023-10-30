import re

OmitText = [
    "[silence]",
    "[noise]",
    "[vocalized-noise]",
]

BACKCHANNELS = [
    "yeah",
    "um-hum",
    "uh-huh",
    "right",
    "oh",
    "oh yeah",
    "yeah yeah",
    "right right",
    "oh really",
    "um-hum um-hum",
    "uh-huh uh-huh",
    "oh uh-huh"
]


def _clean_dialogs():
    pass


def _read_transcript_line(line):
    sepLine = line.split(" ")

    text = " ".join(sepLine[3:]).strip()
    start = float(sepLine[1])
    end = float(sepLine[2])

    return text, start, end


def _return_overlap(textA, textB, startA, startB, endA, endB):
    if startA > startB and endA < endB:
        return textA, 'A'
    elif startB > startA and endB < endA:
        return textB, 'B'

    return None, None


def _check_overlap_silence(past_line, next_line, thresh=1):
    past_text, past_start, past_end = _read_transcript_line(past_line)
    next_text, next_start, next_end = _read_transcript_line(next_line)

    if past_text != '[silence]' or next_text != '[silence]':
        return False

    if past_end - past_start < 1:
        return False

    if next_end - next_start < 1:
        return False

    return True


def read_txt(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
    return lines

# Preprocessing handled by TurnGPT (https://github.com/ErikEkstedt/datasets_turntaking/blob/main/datasets_turntaking/dataset/switchboard/utils.py)


def sub_regex(s):
    """
    Switchboard annotation specific regexp.

    See:
        - `datasets_turntaking/features/dataset/switchboard.md`
        - https://www.isip.piconepress.com/projects/switchboard/doc/transcription_guidelines/transcription_guidelines.pdf

    """
    # Noise
    s = re.sub(r"\[noise\]", "", s)
    s = re.sub(r"\[vocalized-noise\]", "", s)

    # laughter
    s = re.sub(r"\[laughter\]", "", s)
    # laughing and speech e.g. [laughter-yeah] -> yeah
    s = re.sub(r"\[laughter-(\w*)\]", r"\1", s)
    s = re.sub(r"\[laughter-(\w*\'*\w*)\]", r"\1", s)

    # Partial words: w[ent] -> went
    s = re.sub(r"(\w+)\[(\w*\'*\w*)\]", r"\1\2", s)
    # Partial words: -[th]at -> that
    s = re.sub(r"-\[(\w*\'*\w*)\](\w+)", r"\1\2", s)

    # restarts
    s = re.sub(r"(\w+)-\s", r"\1 ", s)
    s = re.sub(r"(\w+)-$", r"\1", s)

    # Pronounciation variants
    s = re.sub(r"(\w+)\_\d", r"\1", s)

    # Mispronounciation [splace/space] -> space
    s = re.sub(r"\[\w+\/(\w+)\]", r"\1", s)

    # Coinage. remove curly brackets... keep word
    s = re.sub(r"\{(\w*)\}", r"\1", s)

    # remove double spacing on last
    s = re.sub(r"\s\s+", " ", s)
    return s.strip()  # remove whitespace start/end


def extract_speaker_timings(transcript, min_word_diff=0.05):
    out = [[], []]
    for speaker in [0, 1]:
        for utterance in transcript[speaker]:
            start, end = utterance["wfeats"][0]["start"], utterance["wfeats"][0]["end"]

            for word in utterance["wfeats"][1:]:
                if word["start"] - end < min_word_diff:
                    end = word["end"]
                else:
                    out[speaker].append((start, end))
                    start = word["start"]
                    end = word["end"]

            out[speaker].append((start, end))
    # print_transcript_timing(transcript, out)
    return out


def print_transcript_timing(dialog, timings):
    dialog = dialog[0]
    timing = timings[0]

    for idx in range(len(dialog)):
        print(dialog[idx])
        print(timing[idx])


def extract_dialog(filenames):
    trans_filenameA, words_filenameA, trans_filenameB, words_filenameB = filenames

    utterancesA = _extract_utterance_word_feats(
        trans_filenameA,
        words_filenameA,
        speaker='A'
    )
    utterancesB = _extract_utterance_word_feats(
        trans_filenameB,
        words_filenameB,
        speaker='B'
    )

    return [utterancesA, utterancesB]


def _extract_word_features(filename, speaker):
    def remove_multiple_whitespace(s):
        s = re.sub(r"\t", " ", s)
        return re.sub(r"\s\s+", " ", s)

    words = read_txt(filename)

    word_feats = {}
    for word_row in words:
        word_row = remove_multiple_whitespace(word_row).strip()

        key, start, end, word = word_row.split(" ")

        # Apply regex?
        word = sub_regex(word)

        # Check if word should be omitted
        if not (word in OmitText or word == ""):
            if key in word_feats:
                word_feats[key].append(
                    {
                        "word": word,
                        "start": float(start),
                        "end": float(end),
                    }
                )
            else:
                word_feats[key] = [{
                    "word": word,
                    "start": float(start),
                    "end": float(end),
                }]
    return word_feats


def _extract_utterance_word_feats(trans_filename, words_filename, speaker):
    word_feats = _extract_word_features(words_filename, speaker)

    transcript = read_txt(trans_filename)

    utterances = []
    for row in transcript:
        key, start, end, *words = row.split(" ")

        if not (words[0] in OmitText and len(words) == 1):
            word_feat = word_feats.get(key, None)

            if word_feat is None:
                continue

            words = " ".join(words)

            # Apply regex?
            words = sub_regex(words)

            utterances.append(
                {
                    "text": words,
                    "wfeats": word_feat,
                    "start": word_feat[0]["start"],
                    "end": word_feat[-1]["end"],
                }
            )
    return utterances


def remove_words_from_dialog(dialog):
    new_dialog = [[], []]
    for speaker in [0, 1]:
        for utterance in dialog[speaker]:
            new_dialog[speaker].append({
                "text": utterance["text"],
                "start": utterance["start"],
                "end": utterance["end"]
            })

    return new_dialog


def combine_dialogue_with_timings(dialogue, timings):
    i1, j1 = 0, 0
    i2, j2 = 0, 0
    curr_speaker = None
    new_dialogue = []
    speaker = []

    curr_word_idxA = 0
    curr_word_idxB = 0

    while i1 < len(timings[0]) and j1 < len(timings[1]):
        startA, endA = timings[0][i1]
        startB, endB = timings[1][j1]

        if startA < startB:
            utterance = _add_dialogue_for_timing(
                dialogue[0][i2], endA, curr_word_idxA)

            if utterance == []:
                i1 += 1
                continue

            utterance['speaker'] = 'A'
            words = utterance['text']

            utterance['text'] = " ".join(words)
            new_dialogue.append(utterance)

            if len(words) + curr_word_idxA == len(dialogue[0][i2]['wfeats']):
                curr_word_idxA = 0
                i2 += 1
            else:
                curr_word_idxA += len(words)
            i1 += 1
        else:
            utterance = _add_dialogue_for_timing(
                dialogue[1][j2], endB, curr_word_idxB)

            if utterance == []:
                j1 += 1
                continue

            utterance['speaker'] = 'B'
            words = utterance['text']

            utterance['text'] = " ".join(words)
            new_dialogue.append(utterance)

            if len(words) + curr_word_idxB == len(dialogue[1][j2]['wfeats']):
                curr_word_idxB = 0
                j2 += 1
            else:
                curr_word_idxB += len(words)

            j1 += 1

    # _pp_dialogue(new_dialogue)
    return new_dialogue, speaker


def _pp_dialogue(dialogue):
    out = ""
    start = 0
    curr_speaker = None

    for idx in range(len(dialogue)):
        if curr_speaker is None or curr_speaker != dialogue[idx]['speaker']:
            curr_speaker = dialogue[idx]['speaker']
            out += f": {start} - {dialogue[idx-1]['end']}"
            start = dialogue[idx]['start']
            out += f"\n{curr_speaker}"
        out += f" {dialogue[idx]['text']}"

    print(out)


def _add_dialogue_for_timing(text, end, curr_idx=0):
    curr = {}

    if curr_idx > len(text['wfeats']):
        return []

    curr['start'] = text['wfeats'][curr_idx]['start']
    curr['text'] = []
    for idx in range(curr_idx, len(text['wfeats'])):
        curr['text'].append(text['wfeats'][idx]['word'])
        if text['wfeats'][idx]['end'] == end:
            curr['end'] = text['wfeats'][idx]['end']
            return curr

    return []


def remove_backchannels(dialogs, speakers):
    last_endA = 0
    last_endB = 0

    backchannelA = None
    backchannelB = None

    output = []

    for idx in range(len(dialogs)):
        if dialogs[idx]['speaker'] == 'A':
            if backchannelA is not None:
                if not _remove_backchannel(backchannelA, dialogs[idx]):
                    output.append(backchannelA)

            backchannelA = _potential_backchannel(last_endA, dialogs[idx])
            if backchannelA is None:
                output.append(dialogs[idx])
            last_endA = dialogs[idx]['end']
        else:
            if backchannelB is not None:
                if not _remove_backchannel(backchannelB, dialogs[idx]):
                    output.append(backchannelB)

            backchannelB = _potential_backchannel(last_endB, dialogs[idx])
            if backchannelB is None:
                output.append(dialogs[idx])
            last_endB = dialogs[idx]['end']

    # _pp_dialogue(dialogs)
    # _pp_dialogue(output)
    return output, speakers


"""
Actually just need to convert into format for parent dataset.
Where in __getitem__(idx) idx refers to the conversation and returns all turns within
a conversation
So this function just needs to return the turn list for a conversation
"""


def combine_consecutive_trps(dialogs):
    combined_dialogs = [dialogs[0]]
    for idx in range(1,len(dialogs)):
        if combined_dialogs[-1]['speaker'] == dialogs[idx]['speaker']:
            combined_dialogs[-1]['text'] += f" {dialogs[idx]['text']}"
            combined_dialogs[-1]['end'] = dialogs[idx]['end']
        else:
            combined_dialogs.append(dialogs[idx])
    return combined_dialogs


def _remove_backchannel(backchannelA, dialog):
    return (dialog['start'] - backchannelA['end']) > 0.5


def _potential_backchannel(last_phrase, current_phrase):
    if (current_phrase['start'] - last_phrase) > 0.5 and current_phrase['text'] in BACKCHANNELS:
        return current_phrase

    return None
