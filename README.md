# coresound_interview

Please see the [Presentation](Presentation1.pdf)


Docker:

docker build -t coresound

Evaluating test set:

cd coresound_interview
python eval.py --model_path 'models/model.pth' --audio_embedding_path 'data/audio_embeddings.pickle' --image_embedding_path 'data/image_embeddings.pickle'
