<script>
    // svelte
    import { onMount } from "svelte";
    import { createEventDispatcher } from "svelte";
    const dispatch = createEventDispatcher();

    // Events
    let predicting = false;
    let training = false;
    let error;

    // Props

    // Data
    let trainingData;

    // Model
    let modelName = "RNNModel";
    let model;

    // Props training
    export let inputSize = 3;

    export let batchSize = 32; // Neuronen min 32 max 512
    export let epochs = 10; // Trainings Epochen 50 iterations
    export let activationFunction = "softmax";
    export let optimizerName = "adam"; // Optimizer
    export let learningRate = 0.011; // Lernrate
    export let neuronCount = 512;
    export let dataLog;

    const activationList = [
        "none",
        "elu",
        "hardSigmoid",
        "linear",
        "relu",
        "relu6",
        "selu",
        "sigmoid",
        "softmax",
        "softplus",
        "softsign",
        "tanh",
        "swish",
        "mish",
    ];

    const optimizerList = [
        "sgd",
        "momentum",
        "adagrad",
        "adadelta",
        "adam",
        "adamax",
        "rmsprop",
    ];

    // lifecycle functions
    onMount(async () => {
        // try {
        //     model = await loadModel(modelName);
        // } catch (error) {
        //     console.warn(error);
        // }
        // if (!model) model = init();
        // model = init();
    });

    // functions

    let init = () => {
        model = create();
        // await compile();
        // await fit();
        // return model;
    };

    let create = () => {
        const model = tf.sequential();
        // let vocabulary_size = 512;
        let vocabulary_size = 21409;
        model.add(
            tf.layers.embedding({
                inputDim: vocabulary_size,
                outputDim: inputSize,
                inputLength: inputSize,
            })
        );
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
        model.add(tf.layers.dense({ units: vocabulary_size, activation: "relu" }));
        model.add(
            tf.layers.dense({ units: vocabulary_size, activation: "softmax" })
        );
        model.add(tf.layers.dense({ units: neuronCount }));

        return model;
    };

    let compile = () => {
        const optimizer = getOptimizer(optimizerName, learningRate);

        debugger;
        // Prepare the model for training.
        return model.compile({
            optimizer: optimizer,
            loss: "categoricalCrossentropy", //tf.losses.meanSquaredError,
            metrics: ["mse"],
        });
    };

    let fit = (inputs, labels) => {
        debugger;
        return model.fit(inputs, labels, {
            batchSize,
            epochs,
            shuffle: true,
            callbacks: [dataLog],
        });
    };

    let save = async (model, name) => {
        return await model.save(`localstorage://${name}`);
    };

    let load = async (name) => {
        return await tf.loadLayersModel(`localstorage://${name}`);
    };

    let prepareData = (data) => {
        //         from keras.preprocessing.text import Tokenizer
        // import nltk
        // from nltk.tokenize import word_tokenize
        // import numpy as np
        // import re
        // from keras.utils import to_categorical
        // from doc3 import training_doc3
        // cleaned = re.sub(r'\W+', ' ', training_doc3).lower()
        // tokens = word_tokenize(cleaned)
        // train_len = 4
        // text_sequences = []
        // for i in range(train_len,len(tokens)):
        //   seq = tokens[i-train_len:i]
        //   text_sequences.append(seq)
        // sequences = {}
        // count = 1
        // for i in range(len(tokens)):
        //   if tokens[i] not in sequences:
        //     sequences[tokens[i]] = count
        //     count += 1
        // tokenizer = Tokenizer()
        // tokenizer.fit_on_texts(text_sequences)
        // sequences = tokenizer.texts_to_sequences(text_sequences)
        // #vocabulary size increased by 1 for the cause of padding
        // vocabulary_size = len(tokenizer.word_counts)+1
        // n_sequences = np.empty([len(sequences),train_len], dtype='int32')
        // for i in range(len(sequences)):
        //   n_sequences[i] = sequences[i]
        // train_inputs = n_sequences[:,:-1]
        // train_targets = n_sequences[:,-1]
        // train_targets = to_categorical(train_targets, num_classes=vocabulary_size)
        // seq_len = train_inputs.shape[1]

        return tf.tidy(() => {
            // data to lower case
            const lowerCaseData = data.toLowerCase();

            // data without special chars
            const cleanData = lowerCaseData.replace(/[^a-zA-Z0-9 ]/g, "");

            // get unique words
            const uniqueWords = [...new Set(cleanData.split(" "))];

            // get vocabulary size
            const vocabularySize = uniqueWords.length;

            // clean data array
            const cleanDataArray = cleanData.split(" ");

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

            let inputs = [];
            let labels = [];

            trainingData.map((e) => {
                inputs.push(e.slice(0, e.length - 1));
                labels.push(e.slice(e.length - 1, e.length));
            });

            // cleanDataArray.map((v, i, a) => {
            //     let helpInputs = [];
            //     let helpLabels = [];
            //     for (let j = 0; j <= inputSize; j++) {
            //         if (j != inputSize) {
            //             helpInputs.push(a[i + j]);
            //         } else {
            //             helpLabels.push(a[i + j]);
            //         }
            //     }
            //     inputs.push(helpInputs);
            //     labels.push(helpLabels);
            // });

            // inputs = inputs.slice(0, 100);
            // labels = labels.slice(0, 100);
            // const inputTensor = tf.tensor3d(null, inputs, [inputs.length, inputSize]);
            // const labelTensor = tf.tensor3d(null, labels, [labels.length, 1]);

            const inputTensor = tf.tensor2d(inputs, [inputs.length, inputSize]);
            const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

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
        const tensorData = prepareData(data);
        const { inputs, labels } = tensorData;

        debugger;
        // create model with new parms
        model = await create();

        // Train the model
        await compile();
        await fit(inputs, labels);

        // saveModel(model, modelName);
        dispatch("training", false);
    }

    // Predict
    export async function predict(inputData, trainingData) {
        dispatch("predicting", true);

        const normalizationData = prepareData(trainingData);
        const { inputMax, inputMin, labelMin, labelMax } = normalizationData;

        const inputs = inputData.map((d) => d.x);
        const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);

        const normalizedInputs = inputTensor
            .sub(inputMin)
            .div(inputMax.sub(inputMin));

        const [x, y] = tf.tidy(() => {
            const ys = model.predict(normalizedInputs);

            const unNormXs = normalizedInputs
                .mul(inputMax.sub(inputMin))
                .add(inputMin);
            const unNormPreds = ys.mul(labelMax.sub(labelMin)).add(labelMin);

            // Un-normalize the data
            return [unNormXs.dataSync(), unNormPreds.dataSync()];
        });

        const predictedPoints = await Array.from(x).map((val, i) => {
            return { x: val, y: y[i] };
        });

        dispatch("predicting", false);
        return predictedPoints;
    }
</script>
