import fastwer
from utils import levenshtein

def extract_music_text(array):
    lines = array.split("\n")
    lyrics = []
    symbols = []
    for idx, l in enumerate(lines):
        if '.\t.\n' in l:
            continue
        if idx > 0 and len(l.rstrip().split('\t')) > 1:
            symbols.append(l.rstrip().split('\t')[0])
            lyrics.append(l.rstrip().split('\t')[1])
 
    return lyrics, symbols, " ".join(lyrics)

def extract_music_textllevel(array):
    lines = []
    lcontent = []
    completecontent = []
    krn = array.split("\n")
    for line in krn:
        line = line.replace("\n", "<b>")
        line = line.split("\t")
        if len(line)>1:
            lcontent.append(line[0])
            completecontent.append(line[0])
            lcontent.append("<t>")
            completecontent.append("<t>")
            for token in line[1]:
                if token != '<':
                    lcontent.append(token)
                    completecontent.append(token)
                else:
                    lcontent.append("<b>")
                    break
        
        lines.append(lcontent)
        lcontent = []
                
    return lines, completecontent

def compute_metrics(predictions_array, gt_array):
    accum_edit_distance_music_u = 0

    accum_edit_disntance_llevel = 0
    accum_edit_disntance_krn = 0

    u_accum_wer = 0
    u_accum_cer = 0


    accum_len_music_u = 0

    accum_len_lines = 0
    accum_len_krn = 0

    predictions = []
    gts = []

    total_samples = len(gt_array)

    for pa, ga in zip(predictions_array, gt_array):
        gt_lines, complete_gt = extract_music_textllevel(ga)
        hyp_lines, complete_hyp = extract_music_textllevel(pa)

        _, gt_music, gt_text_string = extract_music_text(ga)
        _, h_music, h_text_string = extract_music_text(pa)

        predictions.append(h_text_string)
        gts.append(gt_text_string)
        
        u_music_gt = " ".join(gt_music).replace(".", "").split(" ")
        u_music_hyp = " ".join(h_music).replace(".", "").split(" ")

        accum_edit_distance_music_u += levenshtein(u_music_hyp, u_music_gt)

        accum_edit_disntance_llevel += levenshtein(hyp_lines, gt_lines)
        accum_edit_disntance_krn += levenshtein(complete_hyp, complete_gt)

        accum_len_music_u += len(u_music_gt)

        accum_len_lines += len(gt_lines)
        accum_len_krn += len(complete_gt)

        h_unaligned_str = h_text_string.replace(".", "")
        g_unaligned_str = gt_text_string.replace(".", "")

        u_accum_cer += fastwer.score([h_unaligned_str], [g_unaligned_str], char_level=True)
        u_accum_wer += fastwer.score([h_unaligned_str], [g_unaligned_str])

    ser = 100.*accum_edit_distance_music_u / accum_len_music_u
    ler = 100.*accum_edit_disntance_llevel / accum_len_lines
    ker = 100.*accum_edit_disntance_krn / accum_len_krn

    wer = u_accum_wer / total_samples
    cer = u_accum_cer / total_samples


    return ser, cer, wer, ler, ker    
    