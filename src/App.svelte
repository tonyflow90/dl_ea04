<script>
  // ui
  import "smelte/src/tailwind.css";
  import {
    TextField,
    Select,
    Button,
    ProgressCircular,
    List,
    Switch,
  } from "smelte";

  // svelte
  import { onMount } from "svelte";

  import LSTMModel from "./components/LSTMModel.svelte";
  import ConfigUI from "./components/ConfigUI.svelte";

  // Props
  let model;
  let modelName = "LSTM Model";
  let modelIsWorking = false;
  let predicting = false;

  let input = "";
  let inputPrediction = "";
  let predictedItems = [];
  let maxResultSize = 10;
  let useFilter = false;
  let filterText = "";
  let trained = false;

  // initial Config
  let batchSize = 512; // Neuronen min 32 max 512
  let epochs = 10; // Trainings Epochen 50 iterations
  let hiddenLayerCount = 5; // Anzahl der hidden Layer
  let activationFunction = "relu";
  let selectedOptimizer = "adam"; // Optimizer
  let learningRate = 0.01; // Lernrate
  let neuronCount = 50;
  let inputSize = 3;

  // Data
  let trainingDataSets, trainingData, previewText;

  // Documentation
  let mdUrl = "./files/documentation.md";

  // Texts & Labels
  let taskTitle = "Language Model mit RNN";
  let taskNumber = 4;

  let labelSettings = "Settings";
  let labelWordsForPrediction = "words used for predicting";
  let labelShowResults = "Show results";
  let labelFilter = "use filter";
  let labelFilterText = "";

  let labelDatasetsTitle = "Datasets";
  let labelDatasetsText = "Preview Text";

  let labelPredictionInputTitle = "Predicting Input";
  let labelPredictionInput = "Input";
  let labelPredictedItemsTitle = "Predicted Items";

  // workaround smelte Switch on:change doesnt work
  $: if (useFilter) {
    labelFilterText = `(filtered by '${filterText}')`;
    predictInput();
  }

  $: if (!useFilter) {
    predictInput();
  }

  // lifecycle functions
  onMount(async () => {
    let dataset1 = await loadTrainingData(
      "./data/plenarprotokoll_230_20.05.2021.txt"
    );
    let dataset2 = await loadTrainingData("./data/test_data.txt");

    let dataPreview = dataset1.slice(0, 300) + " ...";

    trainingDataSets = [
      {
        value: 0,
        text: `100% (${parseInt(
          dataset1.length
        )} Characters) - Plenarprotokoll 20.05.2021`,
        data: dataset1,
        dataPreview: dataPreview,
      },
      {
        value: 1,
        text: `50% (${parseInt(
          (dataset1.length / 100) * 50
        )} Characters) - Plenarprotokoll 20.05.2021`,
        data: dataset1.slice(0, parseInt((dataset1.length / 100) * 50)),
        dataPreview: dataPreview,
      },
      {
        value: 2,
        text: `25% (${parseInt(
          (dataset1.length / 100) * 25
        )} Characters) - Plenarprotokoll 20.05.2021`,
        data: dataset1.slice(0, parseInt((dataset1.length / 100) * 25)),
        dataPreview: dataPreview,
      },
      {
        value: 3,
        text: `10% (${parseInt(
          (dataset1.length / 100) * 10
        )} Characters) - Plenarprotokoll 20.05.2021`,
        data: dataset1.slice(0, parseInt((dataset1.length / 100) * 10)),
        dataPreview: dataPreview,
      },
      {
        value: 4,
        text: `1% (${parseInt(
          (dataset1.length / 100) * 1
        )} Characters) - Plenarprotokoll 20.05.2021`,
        data: dataset1.slice(0, parseInt((dataset1.length / 100) * 1)),
        dataPreview: dataPreview,
      },
      {
        value: 5,
        text: `Test Data (${parseInt((dataset1.length / 100) * 1)} Characters)`,
        data: dataset2,
        dataPreview: dataset2,
      },
    ];
  });

  // functions
  async function loadTrainingData(url) {
    const dataResponse = await fetch(url);
    const data = await dataResponse.text();
    return data;
  }

  let loadTrainedModel = async () => {
    if (trainingData) {
      await model.load("./model/model.json", trainingData);
      trained = true;
    }
  };

  let train = async () => {
    if (trainingData) {
      await model.train(trainingData);
      trained = true;
    }
  };

  let predict = async (input) => {
    filterText = "";
    let aInput = input.split(" ");
    let result = [];
    if (aInput.length > 3) {
      if (aInput[aInput.length - 1]) {
        filterText = aInput[aInput.length - 1];
      }
      let predictForInputs = aInput.slice(0, aInput.length - 1);
      result = await model.predict(predictForInputs);
    }

    return result;
  };

  let predictInput = async (e) => {
    if (input) {
      predicting = true;
      let results = await predict(input);

      let i = 1;
      predictedItems = results.map((r) => ({
        text: r.word,
        subheading: `position: ${i++} accuracy: ${parseFloat(
          r.acc * 100
        ).toFixed(2)} %`,
      }));

      // filter
      if (filterText && useFilter) {
        predictedItems = predictedItems.filter((p) =>
          p.text.startsWith(filterText)
        );
      }

      // reduce size
      predictedItems = predictedItems.slice(0, maxResultSize);

      predicting = false;
    } else {
      inputPrediction = "";
    }
  };

  let selectItem = async (e) => {
    let bEndsWithSpace = input.endsWith(" ");
    let selectedItem = e.detail;

    if (bEndsWithSpace) {
      input += selectedItem;
    } else {
      let aInput = input.split(" ");
      aInput[aInput.length - 1] = selectedItem;
      input = aInput.join(" ");
    }
    predictedItems = [];
  };
</script>

<header>
  <h5>Einsendeaufgabe {taskNumber}</h5>
  <h3>{taskTitle}</h3>
</header>

<main>
  <LSTMModel
    {modelName}
    {batchSize}
    {inputSize}
    {epochs}
    {selectedOptimizer}
    {learningRate}
    {neuronCount}
    bind:this={model}
    on:predicting={(e) => (modelIsWorking = e.detail)}
    on:training={(e) => (modelIsWorking = e.detail)}
  />

  <div class="grid">
    <div class="settings-grid">
      <div>
        <h5 class="pb-4">{labelSettings}</h5>
        <div style="width:400px">
          <ConfigUI
            disabled={modelIsWorking}
            bind:name={modelName}
            bind:batchSize
            bind:epochs
            bind:hiddenLayerCount
            bind:activationFunction
            bind:selectedOptimizer
            bind:learningRate
            bind:neuronCount
          />
          <TextField
            label={labelWordsForPrediction}
            outlined
            bind:value={inputSize}
          />
        </div>
      </div>
      <div>
        <h5 class="pb-4">{labelDatasetsTitle}</h5>
        <Select
          label={labelDatasetsTitle}
          items={trainingDataSets}
          disabled={modelIsWorking}
          on:change={(v) => {
            trainingData = trainingDataSets[v.detail].data;
            previewText = trainingDataSets[v.detail].dataPreview;
          }}
        />
        <TextField
          label={labelDatasetsText}
          textarea
          rows="10"
          outlined
          disabled
          bind:value={previewText}
        />

        <Button
          block
          outlined
          on:click={train}
          disabled={!trainingData || modelIsWorking}>train</Button
        >

        <br />
        <p style="text-align: center;">OR</p>
        <br />

        <Button
          block
          outlined
          on:click={loadTrainedModel}
          disabled={!trainingData || modelIsWorking}>load trained model</Button
        >
      </div>
    </div>
    {#if modelIsWorking}
      <ProgressCircular />
    {:else}
      <div>
        <h5 class="pt-6 pb-4">{labelPredictionInputTitle}</h5>
        <div>
          <TextField
            label={labelShowResults}
            outlined
            bind:value={maxResultSize}
            on:change={predictInput}
          />
          <Switch
            label="{labelFilter} {labelFilterText}"
            outlined
            bind:value={useFilter}
          />
        </div>
        <TextField
          label={labelPredictionInput}
          bind:value={input}
          on:input={predictInput}
          disabled={!trained}
          outlined
        />
        {#if predicting}
          <ProgressCircular />
        {/if}
        {#if !predicting && predictedItems.length > 0}
          <h7>{labelPredictedItemsTitle}</h7>
          <List items={predictedItems} on:change={selectItem} />
        {/if}
      </div>
    {/if}
    <div>
      <zero-md src={mdUrl} />
    </div>
  </div>
</main>

<footer>
  <div>
    <h5>Ressourcen</h5>
    <a href="https://github.com/tonyflow90/dl_ea03">
      <p>Github Repository</p>
    </a>
    <a href="https://svelte.dev/">
      <p>Svelte</p>
    </a>
    <a href="https://smeltejs.com/">
      <p>Smeltejs</p>
    </a>
    <a href="https://www.bundestag.de/services/opendata">
      <p>Pleanarprotokoll Deutscher Bundestag</p>
    </a>
  </div>
</footer>

<style>
  :root {
    --header-height: 160px;
    --footer-height: 160px;
    min-height: 100vh;
    height: 100vh;
  }

  header {
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    min-height: var(--header-height);
    background-color: var(--color-primary-500);
    color: var(--color-black);
    padding: 1rem;
  }

  main {
    display: flex;
    flex-direction: column;
    justify-content: center;
    padding: 1rem;
  }

  footer {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    min-height: var(--footer-height);
    height: 100%;
    bottom: 0;
    background-color: var(--color-secondary-700);
    text-align: center;
    font-size: small;
    padding: 1rem;
  }

  .grid {
    display: grid;
    grid-template-rows: repeat(auto-fit, minmax(320px, 1fr));
    column-gap: 20px;
    row-gap: 20px;
    justify-items: center;
  }

  .grid * {
    display: flex;
    flex-direction: column;
    width: 100%;
    max-width: 1000px;
  }

  .settings-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    column-gap: 20px;
    row-gap: 20px;
    justify-items: center;
  }
  .settings-grid * {
    display: flex;
    flex-direction: column;
    max-width: 500px;
  }

  @media (min-width: 640px) {
    main {
      max-width: none;
    }
  }
</style>
