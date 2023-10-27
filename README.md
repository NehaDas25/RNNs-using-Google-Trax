# Work Report

## Information

- Name: <ins> DAS, NEHA </ins>
- GitHub: <ins> NehaDas25 </ins>


## Features

- Not Implemented:
  - what features have been implemented

<br><br>

- Implemented:
  - PART 1: Importing the Data
  - PART 1.2: Convert a Line to Tensor
    - Exercise 1: line_to_tensor
      - Implemented a function line_to_tensor() that takes line, EOS_int=1 as inputs.
      - This function returns a list of integers, which will be refered as a tensor.
      -  Used a special integer to represent the end of the sentence (the end of the line) and this will be the EOS_int (end of sentence integer) parameter of the function.
      - This function transforms each character into its unicode integer using ord() function.
      - This passed all the unit test cases.

  - PART 1.3: Batch Generator
    - Exercise 2: data_generator
      - Implemented a function data_generator() that takes batch_size, max_length, data_lines, line_to_tensor=line_to_tensor, shuffle=True as inputs.
      - For this exercise a While True loop is needed which will yield one batch at a time.
      - if index >= num_lines, index is set to 0.
      - The generator should return shuffled batches of data. To achieve this without modifying the actual lines a list containing the indexes of data_lines is created. This list can be shuffled and used to get random batches everytime the index is reset.
      - if len(line) < max_length append line to cur_batch.
       1. Note that a line that has length equal to max_length should not be appended to the batch.
       2. This is because when converting the characters into a tensor of integers, an additional end of sentence token id will be added.So if max_length is 5, and a line has 4 characters, the tensor representing those 4 characters plus the end of sentence character will be of length 5, which is the max length.
      - if len(cur_batch) == batch_size, go over every line, convert it to an int and store it.
      - This passed all the unit-test cases as well.

  - PART 2: Defining the GRU Model
    - Exercise 3: GRULM
      - Implemented a function GRULM() that takes vocab_size=256, d_model=512, n_layers=2, mode='train' as inputs.
      - To implement this model, google's trax package has been used, instead of implementing the GRU from scratch and the necessary methods from a build in package has been provided in the assignment.
      - The following packages when constructing the model has been used here are:
        1. **tl.Serial()**: Combinator that applies layers serially.Here,the layers are passed as arguments to Serial, separated by commas like tl.Serial(tl.Embeddings(...), tl.Mean(...), tl.Dense(...), tl.LogSoftmax(...)).
        2. **tl.ShiftRight()**: Allows the model to go right in the feed forward.ShiftRight(n_shifts=1, mode='train') layer to shift the tensor to the right n_shift times. 
        3. **tl.Embedding(vocab_size, d_feature)**: Initializes the embedding. In this case it is the size of the vocabulary by the dimension of the model.**vocab_size** is the number of unique words in the given vocabulary.**d_feature** is the number of elements in the word embedding (some choices for a word embedding size range from 150 to 300, for example).
        4. **tl.GRU()**: Trax GRU layer.GRU(n_units) builds a traditional GRU of n_cells with dense internal transformations.
        5. **tl.Dense()**: A dense layer. tl.Dense(n_units): The parameter n_units is the number of units chosen for this dense layer.
        6. **tl.LogSoftmax()**: Log of the output probabilities.
      - This passed all the unit-test cases as well.

  - PART 3: Training
  - PART 3.1: Training the Model
    - Exercise 4: train_model
      - Implemented the train_model() to train the neural network that takes model, data_generator, lines, eval_lines, batch_size=32, max_length=64, n_steps=1, output_dir='model/' as inputs.
      - Created a trax.supervised.training.TrainTask object, this encapsulates the aspects of the dataset and the problem at hand:
       1. labeled_data = the labeled data that we want to train on.
       2. loss_fn = tl.CrossEntropyLoss()
       3. optimizer = trax.optimizers.Adam() with learning rate = 0.0005
      - Created a trax.supervised.training.EvalTask object, this encapsulates aspects of evaluating the model:
       1. labeled_data = the labeled data that we want to evaluate on.
       2. metrics = tl.CrossEntropyLoss() and tl.Accuracy()
       3. How frequently we want to evaluate and checkpoint the model.
      - Create a trax.supervised.training.Loop object, this encapsulates the following:
       1. The previously created TrainTask and EvalTask objects.
       2. the training model = GRULM
       3. optionally the evaluation model, if different from the training model. 
      - For bare_train_generator and bare_eval_generator, data_generator(batch_size, max_length, data_lines, shuffle=False) has been used.
      - To iterate it for multiple epochs, itertools.cycle has been used to wrap the data_generator.
      - This passed all the unit-test case as well.

  - PART 4: Evaluation
  - PART 4.1: Evaluating using the Deep Nets
    - Exercise 5: test_model
      - Implemented a function test_model() that takes preds, target as inputs.
      - Returns the log_perplexity of the model.
      - Preds is a tensor of log probabilities.Here,**tl.one_hot()** to transform the target into the same dimension. Then multiply preds and target and find the sum too.
      - Created a mask to only get the non-padded probabilities. 
      - Getting rid of the padding, log_p = log_p * non_pad.
      - Find ths sum and mean using log_p and non_pad.
      - This passed all the unit-test cases as well.



<br><br>

- Partly implemented:
  - model, test_model, model.pkl.gz has not been implemented, it was provided.
  - prediction.npy, target.npy has not been implemented, it was provided.
  - w2_unittest.py was also not implemented as part of assignment to pass all unit-tests for the graded functions().
  - test_support_files was also not implemented as part of assignment to pass all unit-tests for the graded functions().


<br><br>

- Bugs
  - No bugs

<br><br>


## Reflections

- Assignment is very good. Gives a thorough understanding of the basis of Recurrent Neural Network, Vanilla RNN, Gated Recurrent Unit Model, Deep RNN and Cross Entropy Loss function.


## Output

### output:

<pre>
<br/><br/>
Out[3] - 

Number of lines: 125097
Sample line at position 0 A LOVER'S COMPLAINT
Sample line at position 999 With this night's revels and expire the term

Out[4] -

Number of lines: 125097
Sample line at position 0 a lover's complaint
Sample line at position 999 with this night's revels and expire the term

Out[5] -

Number of lines for training: 124097
Number of lines for validation: 1000

Out[6] -

ord('a'): 97
ord('b'): 98
ord('c'): 99
ord(' '): 32
ord('x'): 120
ord('y'): 121
ord('z'): 122
ord('1'): 49
ord('2'): 50
ord('3'): 51

Out[10] - 

[97, 98, 99, 32, 120, 121, 122, 1]
 
Expected Output
[97, 98, 99, 32, 120, 121, 122, 1]

Out[11] - All tests passed

Out[14] - 

(DeviceArray([[49, 50, 51, 52, 53, 54, 55, 56, 57,  1],
              [50, 51, 52, 53, 54, 55, 56, 57, 48,  1]], dtype=int32),
 DeviceArray([[49, 50, 51, 52, 53, 54, 55, 56, 57,  1],
              [50, 51, 52, 53, 54, 55, 56, 57, 48,  1]], dtype=int32),
 DeviceArray([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=int32))

Expected output
(DeviceArray([[49, 50, 51, 52, 53, 54, 55, 56, 57,  1],
              [50, 51, 52, 53, 54, 55, 56, 57, 48,  1]], dtype=int32),
 DeviceArray([[49, 50, 51, 52, 53, 54, 55, 56, 57,  1],
              [50, 51, 52, 53, 54, 55, 56, 57, 48,  1]], dtype=int32),
 DeviceArray([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=int32))

Out[15] - All tests passed

Out[17] - 10

Out[40] - 

Serial[
  Serial[
    ShiftRight(1)
  ]
  Embedding_256_512
  GRU_512
  GRU_512
  Dense_256
  LogSoftmax
]

Expected output
Serial[
  Serial[
    ShiftRight(1)
  ]
  Embedding_256_512
  GRU_512
  GRU_512
  Dense_256
  LogSoftmax
]

Out[41] - All tests passed

Out[45] -

Number of used lines from the dataset: 25881
Batch size (a power of 2): 32
Number of steps to cover one epoch: 808

Expected output:

Number of used lines from the dataset: 25881

Batch size (a power of 2): 32

Number of steps to cover one epoch: 808

Out[47] - 

Step      1: Total number of trainable weights: 3411200
Step      1: Ran 1 train steps in 7.08 secs
Step      1: train CrossEntropyLoss |  5.54524040
Step      1: eval  CrossEntropyLoss |  5.54105139
Step      1: eval          Accuracy |  0.16030833

Out[48] -

Step      1: Total number of trainable weights: 3411200
Step      1: Ran 1 train steps in 8.60 secs
Step      1: train CrossEntropyLoss |  5.54511976
Step      1: eval  CrossEntropyLoss |  5.54281314
Step      1: eval          Accuracy |  0.03820722
 All tests passed

Out[51] -

The log perplexity and perplexity of your model are respectively 1.7646706 5.8396482

Expected Output: The log perplexity and perplexity of your model are respectively around 1.7 and 5.8.

Out[52] - All tests passed

Out[53] - with the briffful dubn? gold,

Out[54] - 

back in your word was py, i lodg
being any upon a his his high fr
the sungent dearful and our mot




<br/><br/>
</pre>
