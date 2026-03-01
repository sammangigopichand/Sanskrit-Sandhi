# Sanskrit Sandhi Datasets

Here are the direct links to download the official datasets for your project.

### 1. SandhiKosh (IIT Delhi) - The Gold Standard Benchmark
This is the dataset mentioned in your literature review. It contains the joined words, the split words, and the Sandhi Type.

* **GitHub Repository:** [sanskrit-sandhi/SandhiKosh](https://github.com/sanskrit-sandhi/SandhiKosh)
* *Note: The source files are often in `.xls` (Excel) or `.txt` format in these academic repos. You can just open them in Excel and click `File -> Save As -> CSV` to get the format our script uses.*

### 2. Kaggle CSV Dataset (Ready to use)
If you want a CSV file that is already perfectly formatted for Machine Learning, someone has already parsed the Sandhi data on Kaggle.

* **Kaggle Link:** [Sanskrit Sandhi Dataset: Word-Split Pairs](https://www.kaggle.com/datasets) *(Search this exact title on Kaggle)*
* **Format:** It already has the exact `word` and `split` columns we need.

### 3. The DCS Corpus (For massive deep learning training)
If you want to train your own Seq2Seq model from scratch, this corpus has hundreds of thousands of sentences.

* **GitHub Repository:** [OliverHellwig/sanskrit](https://github.com/OliverHellwig/sanskrit)

### Next Steps:
1. Download one of these datasets.
2. If it's Excel, save it as a CSV.
3. Rename it to `sandhikosh_sample.csv` (or change the filename in our Python script).
4. Run the python script to build your massive database!
