# VQG4ExpertKnowledge
Official implementation of our paper "Evaluating the Capability of Video Question Generation for Expert-Knowledge Elicitation".

## TODO
- [x] Release the EgoExoAsk QA pairs
- [x] Release the EgoExoAsk benchmark preprocess code and the evaluation code
- [ ] Publish the VQG demo code


## Requirements
We recommend to use a anaconda or docker environment.
```
pip install -r requirements.txt
```

## Preprocess for EgoExoAsk benchmark

EgoExoAsk QA pairs are provided in the `annotations` directory. To construct the EgoExoAsk benchmark used in our paper with the original EgoExo4D dataset, the following preprocess is required. 

1. Download the EgoExo4D dataset following the official document from here https://docs.ego-exo4d-data.org/ . Ensuring the `atomic_descriptions_train/val.json` and `proficiency_demonstration_train/val.json` are downloaded.
2. Run the following commands.
```
python src/video_clips.py
python src/split.py
``` 
This will create a `qa_val_samples_video_w_desc_eval.json` file under the `annotations` directory.

*Note: `src/split.py` will also create the database split `qa_val_samples_video_w_desc_db.json` and the FAISS index file for the RAG method in our original paper. You can ignore these files.*

## Retriever Training
To reproduce the results in our paper, use the default parameters of the script.
```
python src/retriever_train.py \
  --output_path <YOUR_SAVE_PATH>
  --train_file annotations/EgoExoAsk_train.json \
  --eval_file annotations/EgoExoAsk_val.json
```

## VQG results file format
Our evaluation code allows only one question for one clip. Your VQG results should be like this:
```
[
  {
    "video_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
    "annotations": [
      {
        "question": "Why did C switch the manual from their left hand to their right hand?",
        "video": "clips/xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx/00.mp4"
      },
      {
        "question": "Why did C place his left foot on the small yellow foothold instead of the blue one?",
        "video": "clips/xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx/01.mp4"
      },
    ...
    ]
  },
  ...  
]
```
Note that all the questions for each clip are grouped in one compelete video, following the original EgoExo4D dataset.

## Evaluation code
If you have generated a VQG result following the above format, you can evaluate the result using this command.
```
python src/evaluate_retrieval.py \
  --vqg_json <YOUR_FILENAME> \
  --ann_json annotations/qa_val_samples_video_w_desc_eval.json \
  --retriever_model <YOUR_RETRIEVER_PATH> \
  --output_txt <YOUR_FILENAME>.txt \
  --pool_size 50 \
  --recall_ks 1 5 10 \
  --topk_dump 5
```