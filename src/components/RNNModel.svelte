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
    export let batchSize = 32; // Neuronen min 32 max 512
    export let epochs = 10; // Trainings Epochen 50 iterations
    export let hiddenLayerCount = 1; // Anzahl der hidden Layer
    export let minWeight = 0;
    export let maxWeight = 1;
    export let activationFunction = "softmax";
    export let selectedOptimizer = "adam"; // Optimizer
    export let learningRate = 0.001; // Lernrate
    export let neuronCount = 10;
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
        try {
            model = await loadModel(modelName);
        } catch (error) {
            console.warn(error);
        }

        if (!model) model = init();
    });

    // functions

    let init = async () => {
        model = await create(40, 40, [2]);
        await compile();
        await fit();
        return model;
    };

    // def create_model(total_words, hidden_size, num_steps, optimizer='adam'):
    // model = tf.keras.models.Sequential()

    // # Embedding layer / Input layer
    // model.add(tf.keras.layers.Embedding(
    //     total_words, hidden_size, input_length=num_steps))

    // # 4 LSTM layers
    // model.add(tf.keras.layers.LSTM(units=hidden_size, return_sequences=True))
    // model.add(tf.keras.layers.LSTM(units=hidden_size, return_sequences=True))
    // model.add(tf.keras.layers.LSTM(units=hidden_size, return_sequences=True))
    // model.add(tf.keras.layers.LSTM(units=hidden_size, return_sequences=True))

    // # Fully Connected layer
    // model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1024)))
    // model.add(tf.keras.layers.Activation('relu'))
    // model.add(tf.keras.layers.Dropout(0.3, seed=0.2))
    // model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(512)))
    // model.add(tf.keras.layers.Activation('relu'))

    // # Output Layer
    // model.add(tf.keras.layers.TimeDistributed(
    //     tf.keras.layers.Dense(total_words)))
    // model.add(tf.keras.layers.Activation('softmax'))

    // model.compile(loss='categorical_crossentropy', optimizer=optimizer,
    //               metrics=[tf.keras.metrics.categorical_accuracy])
    // return model

    let create = async () => {
        const model = tf.sequential();
        const rnnUnits = 32;
        model.add(tf.layers.simpleRNN({ units: rnnUnits, inputShape }));
        model.add(tf.layers.dense({ units: 1 }));
        return model;

        // let model = tf.sequential();

        // // Add a input layer
        // let inputConfig = {
        //     name: "inputlayer",
        //     inputShape: [1],
        //     units: neuronCount,
        //     useBias: true,
        // };
        // if (activationFunction != "none")
        //     inputConfig.activation = activationFunction;

        // let layer = tf.layers.dense(inputConfig);
        // model.add(layer);

        // // model.add(
        // //     tf.layers.dense({ units: neuronCount, activation: "softmax" })
        // // );

        // return model;
    };
    let compile = async () => {
        const optimizer = getOptimizer(selectedOptimizer, learningRate);

        // Prepare the model for training.
        return model.compile({
            optimizer: optimizer,
            loss: "categoricalCrossentropy", //tf.losses.meanSquaredError,
            metrics: ["mse"],
        });
    };

    let fit = async (inputs, labels) => {
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

        // create model with new parms
        model = createModel();

        // set training data
        trainingData = data;

        // Convert the data to a form we can use for training.
        const tensorData = prepareData(trainingData);
        const { inputs, labels } = tensorData;

        // Train the model
        const optimizer = getOptimizer(selectedOptimizer, learningRate);

        // Prepare the model for training.
        model.compile({
            optimizer: optimizer,
            loss: tf.losses.meanSquaredError,
            metrics: ["mse"],
        });

        await model.fit(inputs, labels, {
            batchSize,
            epochs,
            shuffle: true,
            callbacks: showTraining
                ? tfvis.show.fitCallbacks(
                      chart ? chart : { name: "Training Performance" },
                      ["val_loss", "loss", "val_mse", "mse"],
                      {
                          yAxisDomain: [0, 0.1],
                          height: 200,
                          width: 400,
                          callbacks: ["onEpochEnd"],
                      }
                  )
                : undefined,
        });

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
