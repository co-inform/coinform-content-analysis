import settings
import estimators.utils as utils
import src.sdqc
import torch
from pathlib import Path


class ClearRumorModel:
    model = None

    def __init__(self, model:torch.nn.Model):
        self.model = model
        self.device = utils.get_device()

    def estimate_tweet_stance(self, tweet_id, conversation):
        # calculate post embeddings
        post_embeddings = src.sdqc.util.calculate_post_elmo_embeddings(
            replies,
            max_sentence_length=32,
            batch_size=512,
            scalar_mix_parameters=[0, 0, 0],
            device=torch.device('cpu'))
        # load model
        self.model.eval()
        #p
        return None #p[tweet_id]


if __name__ == '__main__':
    root_dir = Path.cwd().parent
    model = ClearRumorModel(utils.load_model_from_file(root_dir/settings.get_stance_path(ini_path=root_dir/'config.ini')))


env_path = 'data/external/models/sdqc_model_9.pt'
env_path2 = 'data/external/models/verif_model_9.pt'
folder =  'data/external/Replies'

files = [f for f in glob.glob(folder + "/*.jsonl", recursive=True)]

src_file = 'data/external/source/source.jsonl'
replies = {}
for file in files:
    with open(src_file, 'r') as f:
        for line in f:
            src = json.loads(line)
            src_post = ds.Post(id=src['id_str'],
                                 text=src['full_text'],
                                 depth=0,
                                 platform=ds.Post.Platform.twitter,
                                 has_media='media' in src['entities'],
                                 source_id=None,
                                 user_verified=src['user']['verified'],
                                 followers_count=src['user']['followers_count'],
                                 friends_count=src['user']['friends_count'])
            replies[src_post.id] = src_post

# todo depths are buggy



for file in files:
    with open(file, 'r') as f:
        for line in f:
            reply = json.loads(line)
            reply_post = ds.Post(id=reply['id_str'],
                text=reply['full_text'],
                depth=1,
                platform=ds.Post.Platform.twitter,
                has_media='media' in reply['entities'],
                source_id='1176890091253489665',
                user_verified=reply['user']['verified'],
                followers_count=reply['user']['followers_count'],
                friends_count=reply['user']['friends_count'])
            replies[reply_post.id] = reply_post

post_embeddings = src.util.calculate_post_elmo_embeddings(
    replies,
    max_sentence_length=32,
    batch_size=512,
    scalar_mix_parameters=[0, 0, 0],
    device= torch.device('cpu'))
# for i in post_embeddings:
#     post_embeddings[i]["dim"] = 0

# model = sdqc.Sdqc.Model(sdqc_hparams)
model = torch.load(env_path, map_location='cpu')
model.eval()

results = {}
for key, post in replies.items():
    post_features = sdqc.Sdqc.Dataset.calc_features(post, post_embeddings)
    logits = model(post_embeddings[key].unsqueeze(0), torch.from_numpy(post_features).unsqueeze(0))
    probs = F.softmax(logits, dim=1).squeeze().tolist()
    prediction = torch.argmax(logits, dim=1)

    results[key] = \
                (sdqc.SdqcInstance.Label(prediction.item()),
                 dict(zip(sdqc.SdqcInstance.Label, probs)))
