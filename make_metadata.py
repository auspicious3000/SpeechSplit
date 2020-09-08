import os
import pickle
import numpy as np

rootDir = 'assets/spmel'
dirName, subdirList, _ = next(os.walk(rootDir))
print('Found directory: %s' % dirName)


speakers = []
for speaker in sorted(subdirList):
    print('Processing speaker: %s' % speaker)
    utterances = []
    utterances.append(speaker)
    _, _, fileList = next(os.walk(os.path.join(dirName,speaker)))
    
    # use hardcoded onehot embeddings in order to be cosistent with the test speakers
    # modify as needed
    # may use generalized speaker embedding for zero-shot conversion
    spkid = np.zeros((82,), dtype=np.float32)
    if speaker == 'p226':
        spkid[1] = 1.0
    else:
        spkid[7] = 1.0
    utterances.append(spkid)
    
    # create file list
    for fileName in sorted(fileList):
        utterances.append(os.path.join(speaker,fileName))
    speakers.append(utterances)
    
with open(os.path.join(rootDir, 'train.pkl'), 'wb') as handle:
    pickle.dump(speakers, handle)    