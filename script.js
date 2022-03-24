import {TRAINING_DATA} 
from 'https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/fashion-mnist.js';

const CLOTHING_TYPES = ["T-shirt/top", "Trouser", "Pullover", 
                     "Dress", "Coat", "Sandal", "Shirt", 
                     "Sneaker", "Bag", "Ankle boot"];

// Grab a reference to the MNIST input values (pixel data).

const INPUTS = TRAINING_DATA.inputs;

// Grab reference to the MNIST output values.

const OUTPUTS = TRAINING_DATA.outputs;

// Shuffle the two arrays in the same way so inputs still match outputs indexes.

tf.util.shuffleCombo(INPUTS, OUTPUTS);

const INPUTS_TENSOR = normalize(tf.tensor2d(INPUTS), 0, 255);

const OUTPUTS_TENSOR = tf.oneHot(tf.tensor1d(OUTPUTS, 'int32'), 10);

const model = tf.sequential();

model.add(tf.layers.conv2d({

  inputShape: [28, 28, 1],

  filters: 16,

  kernelSize: 3, // Square Filter of 3 by 3. Could also specify rectangle eg [2, 3].

  strides: 1,

  padding: 'same',

  activation: 'relu'  

}));

model.add(tf.layers.maxPooling2d({poolSize: 2, strides: 2}));

model.add(tf.layers.conv2d({

  filters: 32,

  kernelSize: 3,

  strides: 1,

  padding: 'same',

  activation: 'relu'

}));

model.add(tf.layers.maxPooling2d({poolSize: 2, strides: 2}));

model.add(tf.layers.flatten());

model.add(tf.layers.dense({units: 128, activation: 'relu'}));

model.add(tf.layers.dense({units: 10, activation: 'softmax'}));

model.summary();

train();

function normalize(tensor, min, max) {

  const result = tf.tidy(function() {
    
    const MIN_VALUES = min || tf.min(tensor, 0);
    const MAX_VALUES = max || tf.max(tensor, 0);

    const TENSOR_SUBTRACT_MIN_VALUE = tf.sub(tensor, MIN_VALUES);
    const RANGE_SIZE = tf.sub(MAX_VALUES, MIN_VALUES);
    const NORMALIZED_VALUES = tf.div(TENSOR_SUBTRACT_MIN_VALUE, RANGE_SIZE);

    return NORMALIZED_VALUES;

  });

  return result;

}


async function train() { 

  model.compile({

    optimizer: 'adam', // Adam changes the learning rate over time which is useful.

    loss: 'categoricalCrossentropy', // As this is a classification problem, dont use MSE.

    metrics: ['accuracy'] 

  });

  const RESHAPED_INPUTS = INPUTS_TENSOR.reshape([INPUTS.length, 28, 28, 1]);

  let results = await model.fit(RESHAPED_INPUTS, OUTPUTS_TENSOR, {

    shuffle: true,        // Ensure data is shuffled again before using each time.

    validationSplit: 0.15,  

    epochs: 30,           // Go over the data 30 times!

    batchSize: 256,

    callbacks: {onEpochEnd: logProgress}

  });

  RESHAPED_INPUTS.dispose();

  OUTPUTS_TENSOR.dispose();

  INPUTS_TENSOR.dispose();

  evaluate();

}


function logProgress(epoch, logs) {
  
  console.log('Data for epoch ' + epoch, Math.sqrt(logs.loss));
  
}


const PREDICTION_ELEMENT = document.getElementById('prediction');

function evaluate() {

  const OFFSET = Math.floor((Math.random() * INPUTS.length));

  let answer = tf.tidy(function() {

    let newInput = normalize(tf.tensor1d(INPUTS[OFFSET]), 0, 255);

    let output = model.predict(newInput.reshape([1, 28, 28, 1]));

    output.print();

    return output.squeeze().argMax();    

  });

  answer.array().then(function(index) {

    PREDICTION_ELEMENT.innerText = LOOKUP[index];

    PREDICTION_ELEMENT.setAttribute('class', (index === OUTPUTS[OFFSET]) ? 'correct' : 'wrong');

    answer.dispose();

    drawImage(INPUTS[OFFSET]);

  });

}


const CANVAS = document.getElementById('canvas');

const CTX = CANVAS.getContext('2d');


function drawImage(digit) {

  var imageData = CTX.getImageData(0, 0, 28, 28);

  

  for (let i = 0; i < digit.length; i++) {

    imageData.data[i * 4] = digit[i] * 255;      // Red Channel.

    imageData.data[i * 4 + 1] = digit[i] * 255;  // Green Channel.

    imageData.data[i * 4 + 2] = digit[i] * 255;  // Blue Channel.

    imageData.data[i * 4 + 3] = 255;             // Alpha Channel.

  }

  // Render the updated array of data to the canvas itself.

  CTX.putImageData(imageData, 0, 0); 

  // Perform a new classification after a certain interval.

  setTimeout(evaluate, interval);

}


var interval = 2000;

const RANGER = document.getElementById('ranger');

const DOM_SPEED = document.getElementById('domSpeed');

// When user drags slider update interval.

RANGER.addEventListener('input', function(e) {

  interval = this.value;

  DOM_SPEED.innerText = 'Change speed of classification! Currently: ' + interval + 'ms';

});
