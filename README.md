# coresound_interview

    cd coresound_interview

### Running evaluation script
Docker:

    docker build -t coresound . 

    docker run -v $(pwd)/data:/usr/src/app/data -v $(pwd)/models:/usr/src/app/models coresound --audio_embedding_path <audio_embedding_path> --image_embedding_path <image_embedding_path>

Or 

    pip install -r requirements.txt

    python eval.py --model_path 'models/model.pth' --audio_embedding_path 'data/audio_embeddings.pickle' --image_embedding_path 'data/image_embeddings.pickle'

Also please see the [Presentation](Presentation1.pdf)
