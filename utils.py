import numpy as np
from dv3.synthesis import tts as _tts
def generate_cloned_samples(model, cloning_text_path  = None, no_speakers = 108 , fast = True, p =0):
    """
    Generate mel specs with all specker
    Mels that will b feed to the encoder
    return mels for all speckers
    """
    import pickle
    #cloning_texts = ["this is the first" , "this is the second"]
    if(cloning_text_path == None):
        cloning_text_path = "./Cloning_Audio/cloning_text.txt"

    cloning_texts = open(cloning_text_path).read().splitlines()
    # no_cloning_texts = len(cloning_texts)

    all_speakers = []

    for speaker_id in range(no_speakers):
        speaker_cloning_mel = []
        print("The Speaker being cloned speaker-{}".format(speaker_id))
        for text in cloning_texts:
            waveform, alignment, spectrogram, mel = _tts(model, text, p, speaker_id, fast)
            speaker_cloning_mel.append([speaker_id, mel])
        all_speakers.append(speaker_cloning_mel)
        with open("./Cloning_Audio/speakers_cloned_voices_mel.p", "wb") as fp:   #Pickling
            pickle.dump(all_speakers, fp)
        # print("")

    print("Shape of all speakers:",np.array(all_speakers).shape)

    # all speakers[speaker_id][cloned_audio_number]
    return all_speakers

def get_cloned_voices(model, no_speakers = 108,no_cloned_texts = 23):
    import pickle
    try:
        with open("./Cloning_Audio/speakers_cloned_voices_mel.p" , "rb") as fp:
            cloned_voices = pickle.load(fp)
    except:
        print("Generating samples")
        cloned_voices = generate_cloned_samples(model)
        if(np.array(cloned_voices).shape != (no_speakers , no_cloned_texts)):
            cloned_voices = generate_cloned_samples(model,"./Cloning_Audio/cloning_text.txt" ,no_speakers,True,0)
    print("Cloned_voices Loaded!")
    return cloned_voices

def get_embed_speakers(model):
 return np.array(model.embed_speakers.weight.data)