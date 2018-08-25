def preprocess_sotu_text(text):
    removed_applause_text = text.replace('(Applause.)', ' ')
    removed_ddash_text = removed_applause_text.replace('--', ' ')
    processed_text = removed_ddash_text
    return processed_text