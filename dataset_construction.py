import pickle
import random
from unidecode import unidecode

from tqdm import tqdm

def load_embeddings(embedding_path):
    with open(embedding_path, 'rb') as f:
        embeddings = pickle.load(f)
    return embeddings


def extract_speaker_name(path):
    """
    Assuming the format 'SpeakerName/...' in the path,
    and normalizing the speaker name to handle non-English characters.
    """
    return unidecode(path.split('/')[0])

def create_triplets(image_embeddings, audio_embeddings):
    triplets = []
    speakers = list(set(extract_speaker_name(path) for path in audio_embeddings.keys()))
    
    image_embeddings_by_speaker = {speaker: [] for speaker in speakers}
    
    for path, emb in image_embeddings.items():
        speaker_name = extract_speaker_name(path)
        image_embeddings_by_speaker[speaker_name].append(emb)

    for audio_path, audio_emb in tqdm(audio_embeddings.items()):
        speaker_name = extract_speaker_name(audio_path)

        pos_image_embeddings = image_embeddings_by_speaker[speaker_name]
        
        negative_speakers = list(set(speakers) - {speaker_name})
        
        for image_embedding in pos_image_embeddings:
            
            negative_speaker = random.choice(negative_speakers)
            neg_image_embeddings = [emb for path, emb in image_embeddings.items() if extract_speaker_name(path) ==  negative_speaker]
            assert len(neg_image_embeddings) > 0, f'No image embeddings found for speaker {negative_speaker}'
            neg_image_embeddings = random.choice(neg_image_embeddings)

            triplets.append((audio_emb, image_embedding, neg_image_embeddings))
    
    return triplets

def main(image_embeddings_path, audio_embeddings_path, output_path):
    image_embeddings = load_embeddings(image_embeddings_path)
    audio_embeddings = load_embeddings(audio_embeddings_path)
    triplets = create_triplets(image_embeddings, audio_embeddings)

    # Save the triplets to a file
    with open(output_path, 'wb') as f:
        pickle.dump(triplets, f)

if __name__ == '__main__':
    main(image_embeddings_path='data/image_embeddings.pickle', audio_embeddings_path='data/audio_embeddings.pickle', output_path='data/triplets.pickle')
