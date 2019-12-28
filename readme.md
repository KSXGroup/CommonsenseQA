## Commonsense Machine Reading Comprehension

As course project of Deep Learning @ SJTU

### Usage

To train our best performance model, you should:

1. Download the ReCoRD data set from https://sheng-z.github.io/ReCoRD-explorer/.  

2. Put the `train.json` and `dev.json` into `data/dataset/record/` folder

3. Download Sentence Piece tokenizer and ALBERT models and configs from 

   ```
   https://s3.amazonaws.com/models.huggingface.co/bert/albert-base-v2-spiece.model
   https://s3.amazonaws.com/models.huggingface.co/bert/albert-base-v2-pytorch_model.bin
   https://s3.amazonaws.com/models.huggingface.co/bert/albert-base-v2-config.json
   ```

   And put all the file into `ALBERT/pretrained_model` folder. 

4. Then run these instructions to 

   ```
   python3 create_record_data.py --input_file data/dataset/record/train.json --file_name train --type train
   python3 create_record_data.py --input_file data/dataset/record/dev.json --file_name dev --type dev
   ```

   to create the training examples. The training examples are saved to `data/dataset/record/`All data files need about 1.6 GB on your disk.

5. finally run `python3 run_record.py`  to start training.  The training need about 35GB memory and 10 GB GPU  memory.

6. After training, the result is write to `result/predict.json`, you go to `data` directory and run 

   ```
   python3 evaluation.py dataset/record/dev.json ../result/prediction.json
   ```

   to evaluate the prediction result on the develop set.

7. To our best effort, our model's best performance on the ReCoRD dataset is

   ```
   * Exact_match: 58.07
   * F1: 59.75879548229553
   ```

   