<script>
	// ui
	import "smelte/src/tailwind.css";
	import {
		TextField,
		Select,
		Snackbar,
		Button,
		ProgressCircular,
		List,
	} from "smelte";

	// svelte
	import { onMount } from "svelte";
	// import List from "smelte/src/components/List/List.svelte";

	import RNNModel from "./components/RNNModel.svelte";

	// Texts & Labels
	let taskTitle = "Language Model mit RNN";
	let taskNumber = 4;

	let labelPredictionInputTitle = "Predicting Input";
	let labelPredictionInput = "Input";
	let labelPredictedItemsTitle = "Predicted Items";

	let textWaitForModel = "creating/training model";

	// Props
	let model;
	let modelName = "FFNN Model";
	let modelIsWorking = false;
	let predicting = false;

	let input = "";
	let inputPrediction = "";
	let predictedItems = [];

	// Data
	let trainingData;

	// Documentation
	let mdUrl = "./files/documentation.md";

	// lifecycle functions
	onMount(async () => {
		trainingData = await loadTrainingData(
			"./data/test.json"
		);
	});

	// functions
	async function loadTrainingData(url) {
		const dataResponse = await fetch(url);
		const data = await dataResponse.json();
		return data;
	}

	let predict = async (input) => {
		// return await model.predict(input);
		return new Promise((resolve) =>
			setTimeout(() => resolve(input + " test"), 1000)
		);
	};

	let predictInput = async (e) => {
		if (input) {
			predicting = true;
			predictedItems = await predict(input);
			predictedItems = ["t", "s", "x", "q", "w", "e", "r", "t", "z"];
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

	// initial Config
	let batchSize = 100; // Neuronen min 32 max 512
	let epochs = 200; // Trainings Epochen 50 iterations
	let hiddenLayerCount = 5; // Anzahl der hidden Layer
	let activationFunction = "relu";
	let selectedOptimizer = "adam"; // Optimizer
	let learningRate = 0.01; // Lernrate
	let neuronCount = 100;
	let maxWeight = 0;
	let minWeight = 0;
</script>

<header>
	<h5>Einsendeaufgabe {taskNumber}</h5>
	<h3>{taskTitle}</h3>
</header>

<main>
	<RNNModel
		{modelName}
		{batchSize}
		{epochs}
		{minWeight}
		{maxWeight}
		{hiddenLayerCount}
		{activationFunction}
		{selectedOptimizer}
		{learningRate}
		{neuronCount}
		bind:this={model}
		on:predicting={(e) => (modelIsWorking = e.detail)}
		on:training={(e) => (modelIsWorking = e.detail)}
	/>

	<div class="grid">
		{#if modelIsWorking}
			<ProgressCircular />
			<p>{textWaitForModel}</p>
		{:else}
			<div>
				<h5 class="pt-6 pb-4">{labelPredictionInputTitle}</h5>
				<TextField
					label={labelPredictionInput}
					bind:value={input}
					on:input={predictInput}
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

	@media (min-width: 640px) {
		main {
			max-width: none;
		}
	}
</style>
