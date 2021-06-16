<script>
  // svelte
  import { onMount } from "svelte";
  import { createEventDispatcher } from "svelte";
  const dispatch = createEventDispatcher();

  // Props

  // Data
  let trainingData;

  // Model
  let modelName = "LSTMModel";
  let model;

  // Props training
  export let inputSize = 3;

  export let batchSize = 512; // Neuronen min 32 max 512
  export let epochs = 10; // Trainings Epochen 50 iterations
  export let optimizerName = "adam"; // Optimizer
  export let learningRate = 0.01; // Lernrate
  export let neuronCount = 50;
  export let dataLog;

  // lifecycle functions
  onMount(async () => {});

  // functions
  let create = (vocabularySize) => {
    const model = tf.sequential();

    // Embedding layer
    model.add(
      tf.layers.embedding({
        inputDim: vocabularySize,
        outputDim: inputSize,
        inputLength: inputSize,
      })
    );

    // LSTM layers
    model.add(
      tf.layers.lstm({
        units: neuronCount,
        returnSequences: true,
      })
    );

    model.add(
      tf.layers.lstm({
        units: neuronCount,
        returnSequences: false,
      })
    );

    model.add(
      tf.layers.dropout({
        rate: 0.5,
      })
    );

    // Dense layers
    model.add(tf.layers.dense({ units: neuronCount, activation: "relu" }));
    model.add(
      tf.layers.dense({ units: vocabularySize, activation: "softmax" })
    );

    console.log(model.summary());

    return model;
  };

  let compile = () => {
    const optimizer = getOptimizer(optimizerName, learningRate);

    return model.compile({
      optimizer: optimizer,
      loss: "sparseCategoricalCrossentropy",
      metrics: ["accuracy"],
    });
  };

  let fit = (inputs, labels) => {
    return model.fit(inputs, labels, {
      batchSize,
      epochs,
      // callbacks: [dataLog],
      // callbacks: {
      //     onTrainBegin: (logs) => console.log("onTrainBegin:", logs),
      //     onTrainEnd: (logs) => console.log("onTrainEnd:", logs),
      //     onEpochBegin: (epoch, logs) => console.log("onEpochBegin:", epoch, logs),
      //     onEpochEnd: (epoch, logs) => console.log("onEpochEnd:", epoch, logs),
      //     onBatchBegin: (batch, logs) => console.log("onBatchBegin:", batch, logs),
      //     onBatchEnd: (batch, logs) => console.log("onBatchEnd:", batch, logs),
      //     onYield: (epoch, batch, logs) => console.log("onYield:", epoch, batch, logs),
      // },
      callbacks: tfvis.show.fitCallbacks(
        { name: "Training Performance" },
        ["loss", "acc"],
        {
          height: 200,
          width: 400,
          callbacks: ["onBatchEnd", "onEpochEnd"],
        }
      ),
    });
  };

  export async function save(model, name) {
    dispatch("training", true);
    let result = await model.save(`localstorage://${name}`);
    dispatch("training", false);
    return result;
  }

  export async function load(name, data) {
    dispatch("training", true);
    trainingData = prepareData(data);
    model = await tf.loadLayersModel(`${name}`);
    dispatch("training", false);
    return model;
  }

  let prepareData = (data) => {
    return tf.tidy(() => {
      // data to lower case
      const lowerCaseData = data.toLowerCase();

      // data without special chars
      const cleanData = lowerCaseData.replace(/[^a-zA-Z0-9äöüÄÖÜ ]/g, "");

      // get unique words
      const uniqueWords = [...new Set(cleanData.split(" "))];

      // get vocabulary size
      const vocabularySize = uniqueWords.length + 1;

      // clean data array
      let cleanDataArray = cleanData.split(" ");
      cleanDataArray = cleanDataArray.filter((v) => v != " ");

      // clean data array
      let tokenizedDataArray = [];
      cleanDataArray.map((v, i, a) => {
        tokenizedDataArray.push(uniqueWords.indexOf(v));
      });

      let trainingData = [];
      tokenizedDataArray.map((v, i, a) => {
        let help = [];
        for (let j = 0; j <= inputSize; j++) {
          help.push(a[i + j]);
        }
        trainingData.push(help);
      });

      // shuffle trainings data
      // tf.util.shuffle(trainingData);

      let inputs = [];
      let labels = [];

      // trainingData = trainingData.slice(0,85);
      trainingData.map((e) => {
        inputs.push(e.slice(0, e.length - 1));
        let label = e[e.length - 1];
        labels.push(label);
      });

      const inputTensor = tf.tensor(
        inputs,
        [inputs.length, inputSize],
        "int32"
      );
      const labelTensor = tf.tensor(labels, [labels.length, 1], "int32");

      return {
        inputs: inputTensor,
        labels: labelTensor,
        lowerCaseData: lowerCaseData,
        cleanData: cleanData,
        uniqueWords: uniqueWords,
        vocabularySize: vocabularySize,
        originalData: data,
      };
    });
  };

  let getOptimizer = (name, learningRate) => {
    let optimizer;
    switch (name) {
      case "sgd":
        optimizer = tf.train.sgd(learningRate);
        break;
      case "momentum":
        optimizer = tf.train.momentum(learningRate);
        break;
      case "adagrad":
        optimizer = tf.train.adagrad(learningRate);
        break;
      case "adadelta":
        optimizer = tf.train.adadelta(learningRate);
        break;
      case "adam":
        optimizer = tf.train.adam(learningRate);
        break;
      case "adamax":
        optimizer = tf.train.adamax(learningRate);
        break;
      case "rmsprop":
        optimizer = tf.train.rmsprop(learningRate);
        break;
      default:
        optimizer = tf.train.adam(learningRate);
        break;
    }

    return optimizer;
  };

  // Train
  export async function train(data) {
    dispatch("training", true);

    // Convert the data to a form we can use for training.
    trainingData = prepareData(data);
    const { inputs, labels, vocabularySize } = trainingData;

    // create model with new parms
    model = await create(vocabularySize);

    // Train the model
    await compile();
    await fit(inputs, labels);

    // saveModel(model, modelName);
    dispatch("training", false);
  }

  // Predict
  export async function predict(inputData) {
    dispatch("predicting", true);

    inputData = inputData.slice(inputData.length - inputSize, inputData.length);
    let tokenizedInputArray = inputData.map((v) =>
      trainingData.uniqueWords.indexOf(v.toLowerCase())
    );

    const inputs = tf.tensor2d(tokenizedInputArray, [
      1,
      tokenizedInputArray.length,
    ]);

    const [words] = tf.tidy(() => {
      const pwords = model.predict(inputs);
      return [pwords.dataSync()];
    });

    let results = Array.from(words).map((val, i) => {
      return { word: trainingData.uniqueWords[i], acc: val.toFixed(6) };
    });

    let sortedResults = results.sort((w1, w2) => (w1.acc < w2.acc ? 1 : -1));

    dispatch("predicting", false);
    return sortedResults;
  }
</script>
