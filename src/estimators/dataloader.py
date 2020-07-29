import json
import os
import re


tweet_label = {}


def extract_text_and_label(path_annotation: str, path_tweets: str):
    with open(path_annotation) as f:
        data = json.load(f)
        subtask_b = data['subtaskbenglish']

    for d in os.listdir(path_tweets):
        if not os.path.isdir(path_tweets + d):
            continue

        for ff in os.listdir(path_tweets + d):
            if not os.path.isdir(path_tweets + d + '/' + ff):
                continue

            with open(path_tweets + d + '/' + ff + '/source-tweet/' + ff + '.json') as one_tweet:
                tweet_data = json.load(one_tweet)
                text = tweet_data['text']
                if ff not in subtask_b:
                    print(ff + "\n")
                    continue

                tweet_label[ff] = text + '\t' + subtask_b[ff]


def main():
    path_annotation = "/Users/iurshina/co-inform/data/rumoureval2019/rumoureval-2019-training-data/train-key.json"
    path_tweets = "/Users/iurshina/co-inform/data/rumoureval2019/rumoureval-2019-training-data/twitter-english/"

    extract_text_and_label(path_annotation, path_tweets)

    #todo: pheme -- different annotation

    with open("rumeval.tsv", "w") as out:
        for e in tweet_label.keys():
            text = tweet_label[e].replace("\n", "")
            text = re.sub(r"http\S+", '', text)
            out.write(e + "\t" + text + "\n")

    tweet_label.clear()

    path_annotation = "/Users/iurshina/co-inform/data/rumoureval2019/rumoureval-2019-training-data/dev-key.json"
    path_tweets = "/Users/iurshina/co-inform/data/rumoureval2019/rumoureval-2019-training-data/twitter-english/"

    extract_text_and_label(path_annotation, path_tweets)

    with open("rumeval_dev.tsv", "w") as out:
        for e in tweet_label.keys():
            text = tweet_label[e].replace("\n", "")
            text = re.sub(r"http\S+", '', text)
            out.write(e + "\t" + text + "\n")


if __name__ == '__main__':
    main()
