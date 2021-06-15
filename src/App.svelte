<script>
	// ui
	import "smelte/src/tailwind.css";
	import {
		TextField,
		Select,
		Button,
		ProgressCircular,
		List,
	} from "smelte";

	// svelte
	import { onMount } from "svelte";

	import LSTMModel from "./components/LSTMModel.svelte";

	// Texts & Labels
	let taskTitle = "Language Model mit RNN";
	let taskNumber = 4;

	let labelDatasetsTitle = "Datasets";
	let labelDatasetsText = "Preview Text";

	let labelPredictionInputTitle = "Predicting Input";
	let labelPredictionInput = "Input";
	let labelPredictedItemsTitle = "Predicted Items";

	let textWaitForModel = "creating/training model";

	// Props
	let model;
	let modelName = "LSTM Model";
	let modelIsWorking = false;
	let predicting = false;

	let input = "";
	let inputPrediction = "";
	let predictedItems = [];

	// Data
	let trainingDataSets, trainingData, previewText;

	// Documentation
	let mdUrl = "./files/documentation.md";

	// lifecycle functions
	onMount(async () => {
		let dataset1 = await loadTrainingData(
			"./data/plenarprotokoll_230_20.05.2021.txt"
		);

		trainingDataSets = [
			{
				value: 0,
				text: "Plenarprotokoll 20.05.2021",
				data: dataset1,
				dataPreview: dataset1.slice(0, 300) + " ..."
			}, {
				value: 0,
				text: "short version - Plenarprotokoll 20.05.2021",
				data: dataset1.slice(0, 10000),
				dataPreview: dataset1.slice(0, 300) + " ..."
			}
		];
	});

	// functions
	async function loadTrainingData(url) {
		const dataResponse = await fetch(url);
		const data = await dataResponse.text();
		return data;
	}

	let train = async () => {
		if(trainingData)
			model.train(trainingData);
	};

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
</script>

<header>
	<h5>Einsendeaufgabe {taskNumber}</h5>
	<h3>{taskTitle}</h3>
</header>

<main>
	<!-- <RNNModel
		bind:this={model}
		on:predicting={(e) => (modelIsWorking = e.detail)}
		on:training={(e) => (modelIsWorking = e.detail)}
	/> -->

	<LSTMModel
		bind:this={model}
		on:predicting={(e) => (modelIsWorking = e.detail)}
		on:training={(e) => (modelIsWorking = e.detail)}
	/>

	<div class="grid">
		<div>
			<h5 class="pb-4">{labelDatasetsTitle}</h5>
			<Select
				label={labelDatasetsTitle}
				items={trainingDataSets}
				disabled={modelIsWorking}
				on:change={(v) => {
					trainingData = trainingDataSets[v.detail].data;
					previewText = trainingDataSets[v.detail].dataPreview;
					console.log(v.detail)
				}}
			/>
			<TextField label={labelDatasetsText} textarea rows="5" outlined disabled bind:value={previewText} />

			<Button
				block
				outlined
				on:click={train}
				disabled={!trainingData || modelIsWorking}
				>train</Button
			>
		</div>
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

	@media (min-width: 640px) {
		main {
			max-width: none;
		}
	}
</style>
