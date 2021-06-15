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
    let modelName = "LSTMModel";
    let model;

    // Props training
    export let inputSize = 3;

    export let batchSize = 512; // Neuronen min 32 max 512
    export let epochs = 10; // Trainings Epochen 50 iterations
    export let activationFunction = "softmax";
    export let optimizerName = "adam"; // Optimizer
    export let learningRate = 0.001; // Lernrate
    export let neuronCount = 50;
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

    let createOld = (vocabularySize) => {
        const model = tf.sequential();
        // let vocabulary_size = 512;
        // let vocabulary_size = 21410;
        // let vocabulary_size = 182975;
        // model.add(
        //     tf.layers.embedding({
        //         inputDim: vocabularySize,
        //         outputDim: inputSize,
        //         inputLength: inputSize,
        //     })
        // );
        model.add(
            tf.layers.embedding({
                inputDim: 1000,
                outputDim: 200,
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
        // model.add(tf.layers.dense({ units: neuronCount, activation: "relu" }));
        model.add(
            tf.layers.dense({ units: vocabularySize, activation: "softmax" })
        );
        model.add(tf.layers.dense({ units: 1}));
        console.log(model.summary());

        return model;
    };


    let create = (vocabularySize) => {
        const model = tf.sequential();
        model.add(
            tf.layers.embedding({
                inputDim: vocabularySize,
                outputDim: 1,
                inputLength: inputSize,
            })
        );
        model.add(
            tf.layers.lstm({
                units: 200,
                useBias: true,
                returnSequences: true,
            })
        );

        model.add(
            tf.layers.lstm({
                units: 200,
                useBias: true,
                returnSequences: true,
            })
        );

        model.add(
            tf.layers.dropout({
                rate: .5,
            })
        );

        // model.add(
        //     tf.layers.dense({ units: 10000, activation: "softmax" })
        // );

model.add(tf.layers.flatten());

        // model.add(tf.layers.timeDistributed({layer:tf.layers.dense({units:3})}));


        // model.add(tf.layers.dense({ units: neuronCount, activation: "relu" }));
        model.add(
            tf.layers.dense({ units: vocabularySize, activation: "softmax" })
        );
        // model.add(tf.layers.dense({ units: 1}));
        console.log(model.summary());

        return model;
    };

    let compile = () => {
        const optimizer = getOptimizer(optimizerName, learningRate);

        return model.compile({
            optimizer: optimizer,
            // loss: "sparseCategoricalCrossentropy",
            loss: "categoricalCrossentropy", //tf.losses.meanSquaredError,
            metrics: ["accuracy"],
        });
    };

    let fit = (inputs, labels) => {
        return model.fit(inputs, labels, {
            batchSize,
            epochs,
            shuffle: true,
            validationSplit: 0.3,
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
            // onEpochEnd
            callbacks: tfvis.show.fitCallbacks(
                { name: "Training Performance" },
                ["loss", "acc"],
                {
                    height: 200,
                    width: 400,
                    callbacks: ["onBatchEnd"],
                }
            ),
        });
    };

    let save = async (model, name) => {
        return await model.save(`localstorage://${name}`);
    };

    let load = async (name) => {
        return await tf.loadLayersModel(`${name}`);
    };

    let prepareData = (data) => {
        return tf.tidy(() => {
            // data to lower case
            const lowerCaseData = data.toLowerCase();

            // data without special chars
            const cleanData = lowerCaseData.replace(/[^a-zA-Z0-9 ]/g, "");

            // get unique words
            const uniqueWords = [...new Set(cleanData.split(" "))];

            // get vocabulary size
            const vocabularySize = uniqueWords.length + 1;

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

            // shuffle trainings data
            tf.util.shuffle(trainingData);

            let inputs = [];
            let labels = [];

            trainingData.map((e) => {
                inputs.push(e.slice(0, e.length - 1));
                let label = e[e.length - 1];
                labels.push(label);
            });

            // tests
            // inputs = inputs.slice(0, 30000);
            // labels = labels.slice(0, 30000);

            // // tests
            // let inputTensors = [];
            // let labelTensors = [];

            // inputs.forEach((i) => {
            //     inputTensors.push(tf.tensor(i))
            // });

            // labels.forEach((l) => {
            //     labelTensors.push(tf.tensor(l))
            // });

            // const inputTensor = tf.tensor2d(intputTensors, [intputTensors.length, inputSize]);
            // const labelTensor = tf.tensor2d(labelTensors, [labelTensors.length, 1]);

            // inputs = [[[1],[2],[3]],[[4],[5],[6]],[[7],[8],[9]],[[9],[1],[2]]]
            // labels = [4,7,1,3]
            const inputTensor = tf.tensor(inputs, [inputs.length, inputSize],"int32");
            const labelTensor = tf.tensor(labels, [labels.length, 1],"int32");

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
        const { inputs, labels, vocabularySize } = tensorData;

        // create model with new parms

        // model = await create2(inputs.length, vocabularySize, 2);
        model = await create(vocabularySize);
        // model = await load("./model.json");

        // Train the model
        await compile();
        await fit(inputs, labels);

        // saveModel(model, modelName);
        dispatch("training", false);
    }

    // Predict
    export async function predict(inputData, trainingData) {
        dispatch("predicting", true);

        debugger;
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
